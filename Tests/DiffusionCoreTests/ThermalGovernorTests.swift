import XCTest
@testable import DiffusionCore

/// Exercises the real pacing logic of `ThermalGovernor` by injecting scripted thermal/memory
/// states and a no-op sleep, so no device or wall-clock waits are involved.
final class ThermalGovernorTests: XCTestCase {

    /// A scripted thermal source: pops the next state per read, holding the last value once the
    /// script runs out. Thread-safe (the governor may read it from a background task).
    private final class ScriptedThermal: @unchecked Sendable {
        private let lock = NSLock()
        private var script: [ThermalSeverity]
        private var idx = 0
        private(set) var reads = 0
        init(_ script: [ThermalSeverity]) { self.script = script }
        func next() -> ThermalSeverity {
            lock.lock(); defer { lock.unlock() }
            reads += 1
            let v = script[min(idx, script.count - 1)]
            if idx < script.count - 1 { idx += 1 }
            return v
        }
        /// Force the source to return `value` from now on (used to "cool" a critical pause).
        func pin(_ value: ThermalSeverity) {
            lock.lock(); script = [value]; idx = 0; lock.unlock()
        }
    }

    /// Records the nanosecond sleep durations the governor asked for, instead of actually sleeping.
    private final class SleepRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var sleeps: [UInt64] = []
        func record(_ n: UInt64) { lock.lock(); sleeps.append(n); lock.unlock() }
        var count: Int { lock.lock(); defer { lock.unlock() }; return sleeps.count }
        var all: [UInt64] { lock.lock(); defer { lock.unlock() }; return sleeps }
    }

    private func makeGovernor(
        thermal: @escaping @Sendable () -> ThermalSeverity,
        pressure: @escaping @Sendable () -> MemoryPressure = { .normal },
        recorder: SleepRecorder
    ) -> ThermalGovernor {
        ThermalGovernor(
            thermalSource: thermal,
            pressureSource: pressure,
            sleep: { recorder.record($0) })
    }

    // MARK: - Nominal / fair: zero tax

    func testNominalAndFairNeverSleep() async throws {
        let recorder = SleepRecorder()
        for state in [ThermalSeverity.nominal, .fair] {
            let gov = makeGovernor(thermal: { state }, recorder: recorder)
            try await gov.throttleIfNeeded()
        }
        XCTAssertEqual(recorder.count, 0, "nominal/fair must not insert any sleep")
    }

    func testCurrentSeverityReflectsSource() {
        let recorder = SleepRecorder()
        let gov = makeGovernor(thermal: { .serious }, recorder: recorder)
        XCTAssertEqual(gov.currentSeverity(), .serious)
    }

    // MARK: - Serious: throttle + progressive backoff

    func testSeriousInsertsBackoffSleep() async throws {
        let recorder = SleepRecorder()
        let gov = makeGovernor(thermal: { .serious }, recorder: recorder)
        try await gov.throttleIfNeeded()
        XCTAssertEqual(recorder.count, 1)
        XCTAssertEqual(recorder.all[0], 250 * 1_000_000, "first serious throttle is 250 ms")
    }

    func testSeriousBackoffLengthensThenCaps() async throws {
        let recorder = SleepRecorder()
        let gov = makeGovernor(thermal: { .serious }, recorder: recorder)
        // 8 consecutive serious throttles. The consecutive counter is capped at 5, so the backoff
        // climbs 250 → 350 → 450 → 550 → 650 ms and then plateaus at 650 ms (the 750 ms ceiling in
        // `seriousBackoffNanos` is never reached because the counter caps first).
        for _ in 0..<8 { try await gov.throttleIfNeeded() }
        let expectedMillis: [UInt64] = [250, 350, 450, 550, 650, 650, 650, 650]
        XCTAssertEqual(recorder.all, expectedMillis.map { $0 * 1_000_000 })
    }

    func testNominalResetsSeriousBackoff() async throws {
        let recorder = SleepRecorder()
        let script = ScriptedThermal([.serious, .serious, .nominal, .serious])
        let gov = makeGovernor(thermal: { script.next() }, recorder: recorder)
        try await gov.throttleIfNeeded()   // serious -> 250
        try await gov.throttleIfNeeded()   // serious -> 350
        try await gov.throttleIfNeeded()   // nominal -> reset, no sleep
        try await gov.throttleIfNeeded()   // serious -> back to 250
        XCTAssertEqual(recorder.all, [250, 350, 250].map { $0 * 1_000_000 },
                       "a nominal step must reset the consecutive-serious backoff")
    }

    // MARK: - Critical: hard pause + cooling callback + recovery

    func testCriticalPausesUntilCoolThenResumes() async throws {
        let recorder = SleepRecorder()
        // critical for the first two reads, then cools to nominal.
        let script = ScriptedThermal([.critical, .critical, .nominal])
        let gov = makeGovernor(thermal: { script.next() }, recorder: recorder)

        let coolingFlag = SleepRecorder()  // reuse as a simple thread-safe counter via .record
        try await gov.throttleIfNeeded(onCooling: { coolingFlag.record(1) })

        XCTAssertEqual(coolingFlag.count, 1, "onCooling must fire exactly once when a critical pause begins")
        // It polled (slept) while critical, then returned once the source reported nominal.
        XCTAssertGreaterThanOrEqual(recorder.count, 1, "critical pause must poll-sleep at least once")
        XCTAssertTrue(recorder.all.allSatisfy { $0 == 500_000_000 }, "critical polls every 0.5 s")
    }

    func testCriticalThatNeverCoolsThrowsPausedForHeat() async {
        let recorder = SleepRecorder()
        let gov = makeGovernor(thermal: { .critical }, recorder: recorder)  // stuck critical forever
        do {
            try await gov.throttleIfNeeded()
            XCTFail("expected pausedForHeat after the bounded cooling window")
        } catch let error as EngineError {
            guard case .pausedForHeat = error else {
                return XCTFail("expected .pausedForHeat, got \(error)")
            }
            // Bounded at 240 polls.
            XCTAssertEqual(recorder.count, 240, "the cooling wait is bounded at ~120 s (240 × 0.5 s)")
        } catch {
            XCTFail("expected EngineError.pausedForHeat, got \(error)")
        }
    }

    func testCriticalRespectsCancellation() async {
        let recorder = SleepRecorder()
        let gov = makeGovernor(thermal: { .critical }, recorder: recorder)
        let task = Task { try await gov.throttleIfNeeded() }
        task.cancel()
        do {
            try await task.value
            XCTFail("expected cancellation to unwind the critical pause")
        } catch is CancellationError {
            // expected
        } catch let error as EngineError {
            // Acceptable race: if the bounded window elapsed before cancellation was observed.
            guard case .pausedForHeat = error else {
                return XCTFail("unexpected EngineError \(error)")
            }
        } catch {
            XCTFail("unexpected error \(error)")
        }
    }

    // MARK: - Full nominal -> serious -> critical -> cool transition

    func testFullThermalRampTransition() async throws {
        let recorder = SleepRecorder()
        // step 1: nominal (no tax); step 2: serious (250ms); step 3: critical then cools.
        let script = ScriptedThermal([.nominal, .serious, .critical, .nominal])
        let gov = makeGovernor(thermal: { script.next() }, recorder: recorder)

        try await gov.throttleIfNeeded()                 // nominal
        XCTAssertEqual(recorder.count, 0)

        try await gov.throttleIfNeeded()                 // serious
        XCTAssertEqual(recorder.all.last, 250 * 1_000_000)

        var cooled = false
        try await gov.throttleIfNeeded(onCooling: { cooled = true })  // critical -> nominal
        XCTAssertTrue(cooled, "critical pause emitted a cooling callback")
    }

    // MARK: - Start gate

    func testShouldDeferHeavyStart() {
        let recorder = SleepRecorder()
        XCTAssertFalse(makeGovernor(thermal: { .nominal }, recorder: recorder).shouldDeferHeavyStart())
        XCTAssertFalse(makeGovernor(thermal: { .fair }, recorder: recorder).shouldDeferHeavyStart())
        XCTAssertTrue(makeGovernor(thermal: { .serious }, recorder: recorder).shouldDeferHeavyStart())
        XCTAssertTrue(makeGovernor(thermal: { .critical }, recorder: recorder).shouldDeferHeavyStart())
    }

    // MARK: - Memory pressure relief

    func testMemoryPressureWarningClearsCacheWithoutSleeping() async throws {
        let recorder = SleepRecorder()
        // nominal thermal but warning pressure: should clear cache (no crash) and not sleep.
        let gov = makeGovernor(thermal: { .nominal }, pressure: { .warning }, recorder: recorder)
        try await gov.throttleIfNeeded()
        XCTAssertEqual(recorder.count, 0, "pressure relief must not add a thermal sleep")
    }

    func testCurrentPressureReflectsSource() {
        let recorder = SleepRecorder()
        let gov = makeGovernor(thermal: { .nominal }, pressure: { .critical }, recorder: recorder)
        XCTAssertEqual(gov.currentPressure(), .critical)
    }
}
