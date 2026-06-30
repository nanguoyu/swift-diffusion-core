@preconcurrency import MLX
import Foundation

/// Thermal severity, a Sendable mirror of `ProcessInfo.ThermalState`.
public enum ThermalSeverity: Int, Sendable, Comparable {
    case nominal = 0, fair, serious, critical
    public static func < (a: ThermalSeverity, b: ThermalSeverity) -> Bool { a.rawValue < b.rawValue }
}

/// Memory-pressure level reported by the dispatch memory-pressure source.
public enum MemoryPressure: Int, Sendable, Comparable {
    case normal = 0, warning, critical
    public static func < (a: MemoryPressure, b: MemoryPressure) -> Bool { a.rawValue < b.rawValue }
}

/// On-device thermal & memory-pressure governor — the single hard guarantee against a thermal
/// shutdown while generating.
///
/// MLX exposes no GPU-clock or QoS knob (only memory-side levers), so the ONLY way to shed heat is
/// to stop feeding the GPU. The governor reads `ProcessInfo.thermalState` (cached, updated by the
/// system notification so the hot loop never makes a syscall) and, between denoise steps:
///   • `.nominal` / `.fair`  → returns immediately (zero tax in the common case);
///   • `.serious`            → clears the MLX reuse pool and inserts a short cooperative sleep,
///                             lowering the *duty cycle* so the SoC junction temperature climbs
///                             more slowly (this does NOT lower energy/work — it trades wall-clock
///                             for a gentler thermal slope so the `.critical` wall is reached later
///                             or never);
///   • `.critical`           → PAUSES the loop until the device cools, surfacing a non-error
///                             "cooling" progress; after a bounded timeout it throws the recoverable
///                             `EngineError.pausedForHeat` rather than letting iOS thermal-shutdown
///                             the phone.
///
/// It additionally gates the START of a heavy run (`shouldDeferHeavyStart`): a fresh 1024 request
/// kicked off on an already-warm phone has no headroom and the first `.serious` flip arrives too
/// late in a short denoise to help, so the engine/router refuses or defers it up front.
///
/// On macOS the governor is a runtime no-op: the default thermal source reports `.nominal` and the
/// memory source reports `.normal`, so `throttleIfNeeded` falls straight through the `.nominal`
/// branch — no sleep, no cache clear — keeping the byte-for-byte Mac path untouched.
///
/// ## Testability
/// The thermal-state and memory-pressure inputs, plus the cooperative sleep, are injected. The
/// production initializer wires them to the live system sources (and, on iOS, a notification +
/// dispatch memory-pressure source that keep the cache warm). A test can construct a governor with
/// a scripted `thermalSource` / `pressureSource` and a no-op `sleep` to exercise the real pacing
/// logic deterministically without a device or wall-clock waits.
public final class ThermalGovernor: @unchecked Sendable {
    public static let shared = ThermalGovernor()

    public typealias ThermalSource = @Sendable () -> ThermalSeverity
    public typealias PressureSource = @Sendable () -> MemoryPressure
    public typealias SleepFn = @Sendable (UInt64) async throws -> Void

    /// Reads the current thermal severity. Re-read on demand (start of a throttle, each cooling
    /// poll) so an injected script flows through; in production this returns the notification-warmed
    /// cache on iOS and a constant `.nominal` on macOS.
    private let thermalSource: ThermalSource
    /// Reads the current memory-pressure level.
    private let pressureSource: PressureSource
    /// Cooperative, cancellation-aware sleep. Defaults to `Task.sleep`; tests inject a no-op.
    private let sleepFn: SleepFn

    private let lock = NSLock()
    private var _severity: ThermalSeverity = .nominal
    private var _pressure: MemoryPressure = .normal
    /// How many consecutive `.serious` throttles we've inserted — used to lengthen the backoff the
    /// longer the device stays hot.
    private var consecutiveSeriousThrottles = 0

    #if os(iOS)
    private var thermalObserver: NSObjectProtocol?
    private var memorySource: DispatchSourceMemoryPressure?
    #endif

    /// Production initializer — wires the live system thermal/memory sources and `Task.sleep`.
    public convenience init() {
        #if os(iOS)
        // The cache is kept warm by the notification observer below; the source reads it.
        let box = SeverityBox(.nominal)
        let pressureBox = PressureBox(.normal)
        self.init(
            thermalSource: { box.value },
            pressureSource: { pressureBox.value },
            sleep: { try await Task.sleep(nanoseconds: $0) })
        box.value = Self.map(ProcessInfo.processInfo.thermalState)
        _severity = box.value
        thermalObserver = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: nil
        ) { _ in
            box.value = Self.map(ProcessInfo.processInfo.thermalState)
        }
        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical], queue: .global(qos: .utility))
        source.setEventHandler { [weak source] in
            guard let source else { return }
            // Record only; never touch MLX from the dispatch thread (the GPU may be mid-eval on the
            // generation task). The next `throttleIfNeeded` clears the cache on the generation task.
            pressureBox.value = source.data.contains(.critical) ? .critical : .warning
        }
        source.resume()
        memorySource = source
        #else
        // macOS: plugged in, always nominal — a true runtime no-op.
        self.init(
            thermalSource: { .nominal },
            pressureSource: { .normal },
            sleep: { try await Task.sleep(nanoseconds: $0) })
        #endif
    }

    /// Designated initializer with injectable inputs. Use directly in tests to script the device's
    /// thermal/memory state and bypass real sleeps.
    public init(thermalSource: @escaping ThermalSource,
                pressureSource: @escaping PressureSource = { .normal },
                sleep: @escaping SleepFn = { try await Task.sleep(nanoseconds: $0) }) {
        self.thermalSource = thermalSource
        self.pressureSource = pressureSource
        self.sleepFn = sleep
        // The cache is refreshed on every `currentSeverity()` / `currentPressure()` call, so it is
        // seeded lazily rather than read here — this keeps construction side-effect-free for sources
        // that are stateful (e.g. a scripted test sequence) or expensive to read.
    }

    deinit {
        #if os(iOS)
        if let thermalObserver { NotificationCenter.default.removeObserver(thermalObserver) }
        memorySource?.cancel()
        #endif
    }

    // MARK: - Cached state

    /// Latest thermal severity, re-read from the source. `.nominal` on macOS.
    public func currentSeverity() -> ThermalSeverity {
        let s = thermalSource()
        lock.lock(); _severity = s; lock.unlock()
        return s
    }

    /// Latest memory-pressure level, re-read from the source. `.normal` on macOS.
    public func currentPressure() -> MemoryPressure {
        let p = pressureSource()
        lock.lock(); _pressure = p; lock.unlock()
        return p
    }

    // MARK: - Start gate

    /// True when it is unsafe to BEGIN a heavy (e.g. 1024) run right now because the device is
    /// already hot. The caller should defer the run with a "cooling" message or fall back to a
    /// lighter resolution. Always `false` on macOS (source reports `.nominal`).
    public func shouldDeferHeavyStart() -> Bool {
        return currentSeverity() >= .serious
    }

    // MARK: - Per-step throttle

    /// Pace the denoise loop according to the current thermal state. Call once per step (or, for a
    /// streaming loop running very hot, per block). Cooperative and cancellation-aware: the sleeps
    /// are `Task.sleep`, so a cancelled generation unwinds promptly.
    ///
    /// On macOS this is a fall-through no-op (`.nominal` source): no sleep, no cache clear.
    ///
    /// - Parameter onCooling: invoked (once, when a `.critical` pause begins) so the engine can emit
    ///   a non-error "cooling" progress to the UI.
    /// - Throws: `CancellationError` if the task is cancelled while throttling; `EngineError
    ///   .pausedForHeat` if the device stays `.critical` past the bounded cooling timeout.
    public func throttleIfNeeded(onCooling: (@Sendable () -> Void)? = nil) async throws {
        // Relieve memory pressure on the generation task (never from the dispatch handler).
        if currentPressure() >= .warning { MLX.GPU.clearCache() }

        switch currentSeverity() {
        case .nominal, .fair:
            resetSeriousBackoff()
            return

        case .serious:
            MLX.GPU.clearCache()
            try await sleepFn(seriousBackoffNanos())

        case .critical:
            try await pauseUntilCool(onCooling: onCooling)
        }
    }

    // MARK: - Pacing logic

    /// Backoff at `.serious`: 250 ms, lengthening 100 ms per consecutive serious throttle up to
    /// 750 ms, so a phone that stays hot opens progressively wider GPU-idle gaps.
    private func seriousBackoffNanos() -> UInt64 {
        lock.lock()
        consecutiveSeriousThrottles = min(consecutiveSeriousThrottles + 1, 5)
        let steps = consecutiveSeriousThrottles
        lock.unlock()
        let millis = 250 + 100 * (steps - 1)            // 250…650 (+ the cap below)
        return UInt64(min(millis, 750)) * 1_000_000
    }

    private func resetSeriousBackoff() {
        lock.lock(); consecutiveSeriousThrottles = 0; lock.unlock()
    }

    /// Hold the loop until the device drops out of `.critical`, polling its current state. Bounded so
    /// a phone that simply won't cool surfaces a recoverable error instead of hanging the run.
    private func pauseUntilCool(onCooling: (@Sendable () -> Void)?) async throws {
        MLX.GPU.clearCache()
        onCooling?()
        let pollNanos: UInt64 = 500_000_000                 // 0.5 s
        let maxPolls = 240                                  // up to ~120 s of cooling
        var polls = 0
        while currentSeverity() >= .critical {
            try Task.checkCancellation()
            if polls >= maxPolls { throw EngineError.pausedForHeat }
            try await sleepFn(pollNanos)
            polls += 1
        }
        resetSeriousBackoff()
    }

    // MARK: - Helpers

    #if os(iOS)
    private static func map(_ state: ProcessInfo.ThermalState) -> ThermalSeverity {
        switch state {
        case .nominal:  return .nominal
        case .fair:     return .fair
        case .serious:  return .serious
        case .critical: return .critical
        @unknown default: return .serious
        }
    }
    #endif
}

#if os(iOS)
/// A tiny lock-guarded box so the notification/dispatch handlers can publish state into the
/// closures the governor reads, without capturing `self` (avoids retain cycles on the singleton).
private final class SeverityBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: ThermalSeverity
    init(_ v: ThermalSeverity) { _value = v }
    var value: ThermalSeverity {
        get { lock.lock(); defer { lock.unlock() }; return _value }
        set { lock.lock(); _value = newValue; lock.unlock() }
    }
}

private final class PressureBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: MemoryPressure
    init(_ v: MemoryPressure) { _value = v }
    var value: MemoryPressure {
        get { lock.lock(); defer { lock.unlock() }; return _value }
        set { lock.lock(); _value = newValue; lock.unlock() }
    }
}
#endif
