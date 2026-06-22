/// How a model+variant will run on a specific device — the basis for the gallery's
/// hardware-fit badge and the engine's load plan.
public struct EngineCapabilities: Sendable {
    /// Whether it can run at all on this device.
    public var runnable: Bool
    /// How it will be loaded.
    public var residency: Residency
    /// Estimated peak working set, bytes.
    public var estimatedPeakBytes: Int64
    /// One-line, user-facing rationale (e.g. "Streams from SSD").
    public var note: String

    public init(runnable: Bool, residency: Residency, estimatedPeakBytes: Int64, note: String) {
        self.runnable = runnable
        self.residency = residency
        self.estimatedPeakBytes = estimatedPeakBytes
        self.note = note
    }

    public enum Residency: Sendable {
        /// Whole model held in memory — fastest.
        case resident
        /// Two-phase staging (encoder released before transformer). Tight but resident.
        case twoPhase
        /// Denoiser streamed block-by-block from internal storage.
        case streamingInternal
        /// Streamed from an external USB-C SSD.
        case streamingExternal
        /// Does not fit this device.
        case unsupported
    }
}
