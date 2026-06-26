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
/// On macOS every method is a compile-time no-op: Macs are plugged in, `thermalState` is `.nominal`,
/// and the byte-for-byte Mac path must never sleep or clear caches mid-run.
public final class ThermalGovernor: @unchecked Sendable {
    public static let shared = ThermalGovernor()

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

    public init() {
        #if os(iOS)
        _severity = Self.map(ProcessInfo.processInfo.thermalState)
        thermalObserver = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: nil
        ) { [weak self] _ in
            self?.setSeverity(Self.map(ProcessInfo.processInfo.thermalState))
        }
        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical], queue: .global(qos: .utility))
        source.setEventHandler { [weak self] in
            guard let self, let source = self.memorySource else { return }
            // Record only; never touch MLX from the dispatch thread (the GPU may be mid-eval on the
            // generation task). The next `throttleIfNeeded` clears the cache on the generation task.
            self.setPressure(source.data.contains(.critical) ? .critical : .warning)
        }
        source.resume()
        memorySource = source
        #endif
    }

    deinit {
        #if os(iOS)
        if let thermalObserver { NotificationCenter.default.removeObserver(thermalObserver) }
        memorySource?.cancel()
        #endif
    }

    // MARK: - Cached state

    /// Latest cached thermal severity (no syscall). `.nominal` on macOS.
    public func currentSeverity() -> ThermalSeverity {
        lock.lock(); defer { lock.unlock() }
        return _severity
    }

    /// Latest cached memory-pressure level. `.normal` on macOS.
    public func currentPressure() -> MemoryPressure {
        lock.lock(); defer { lock.unlock() }
        return _pressure
    }

    private func setSeverity(_ s: ThermalSeverity) {
        lock.lock(); _severity = s; lock.unlock()
    }

    private func setPressure(_ p: MemoryPressure) {
        lock.lock(); _pressure = p; lock.unlock()
    }

    // MARK: - Start gate

    /// True when it is unsafe to BEGIN a heavy (e.g. 1024) run right now because the device is
    /// already hot. The caller should defer the run with a "cooling" message or fall back to a
    /// lighter resolution. Always `false` on macOS.
    public func shouldDeferHeavyStart() -> Bool {
        #if os(iOS)
        return currentSeverity() >= .serious
        #else
        return false
        #endif
    }

    // MARK: - Per-step throttle

    /// Pace the denoise loop according to the current thermal state. Call once per step (or, for a
    /// streaming loop running very hot, per block). Cooperative and cancellation-aware: the sleeps
    /// are `Task.sleep`, so a cancelled generation unwinds promptly.
    ///
    /// - Parameter onCooling: invoked (once, when a `.critical` pause begins) so the engine can emit
    ///   a non-error "cooling" progress to the UI.
    /// - Throws: `CancellationError` if the task is cancelled while throttling; `EngineError
    ///   .pausedForHeat` if the device stays `.critical` past the bounded cooling timeout.
    public func throttleIfNeeded(onCooling: (@Sendable () -> Void)? = nil) async throws {
        #if os(iOS)
        // Relieve memory pressure on the generation task (never from the dispatch handler).
        if currentPressure() >= .warning { MLX.GPU.clearCache() }

        switch currentSeverity() {
        case .nominal, .fair:
            resetSeriousBackoff()
            return

        case .serious:
            MLX.GPU.clearCache()
            try await Task.sleep(nanoseconds: seriousBackoffNanos())

        case .critical:
            try await pauseUntilCool(onCooling: onCooling)
        }
        #else
        _ = onCooling
        #endif
    }

    #if os(iOS)
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

    /// Hold the loop until the device drops out of `.critical`, polling its cached state. Bounded so
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
            try await Task.sleep(nanoseconds: pollNanos)
            polls += 1
        }
        resetSeriousBackoff()
    }
    #endif

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
