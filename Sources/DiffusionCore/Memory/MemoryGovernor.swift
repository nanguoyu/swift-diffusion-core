import Foundation

/// Decides *how* a model+variant runs on a device — the rung of the partial-load ladder —
/// and produces the `EngineCapabilities` the gallery shows and the engine obeys.
public enum MemoryGovernor {

    /// Rough activation/latent/attention working set on top of resident weights.
    private static let workingSet: Int64 = 800_000_000

    /// Plan residency for a variant. `externalSSDAvailable` allows the streaming-external rung
    /// when the model is otherwise too big for memory.
    public static func plan(variant: ModelVariant, device: DeviceTier,
                            externalSSDAvailable: Bool) -> EngineCapabilities {
        let c = variant.components
        let budget = device.memoryBudgetBytes

        // Two-phase peak: encoder phase vs. transformer+VAE phase never co-reside.
        let twoPhasePeak = max(c.textEncoder, c.transformer + c.vae) + workingSet
        // Streaming peak: only the encoder (run once) + a few resident blocks + working set.
        let streamingPeak = max(c.textEncoder, 1_200_000_000) + workingSet

        if twoPhasePeak <= Int64(Double(budget) * 0.9) {
            return EngineCapabilities(runnable: true, residency: .resident,
                                      estimatedPeakBytes: twoPhasePeak, note: "Runs great")
        }
        if twoPhasePeak <= Int64(Double(budget) * 1.12) {
            return EngineCapabilities(runnable: true, residency: .twoPhase,
                                      estimatedPeakBytes: twoPhasePeak, note: "Tight · two-phase")
        }
        if streamingPeak <= Int64(Double(budget) * 0.98) {
            let external = device.isPhone && externalSSDAvailable
            return EngineCapabilities(
                runnable: true,
                residency: external ? .streamingExternal : .streamingInternal,
                estimatedPeakBytes: streamingPeak,
                note: external ? "Streams from SSD" : "Streams from disk")
        }
        // Rejection is driven by streamingPeak (the true minimum), so report that honest number.
        return EngineCapabilities(runnable: false, residency: .unsupported,
                                  estimatedPeakBytes: streamingPeak, note: "Needs more memory")
    }

    /// Live headroom for this process (bytes). Use this — not nominal RAM — at load time.
    /// Returns `Int64.max` where the probe doesn't apply (avoids a `UInt64.max → Int64` trap).
    public static func availableBytesNow() -> Int64 {
        let bytes = MemoryProbe.availableBytes()
        return bytes == .max ? .max : Int64(bytes)
    }

    /// The next-leaner residency rung — used when live headroom is below the planned peak.
    public static func leaner(than residency: EngineCapabilities.Residency) -> EngineCapabilities.Residency {
        switch residency {
        case .resident: return .twoPhase
        case .twoPhase: return .streamingInternal
        case .streamingInternal, .streamingExternal, .unsupported: return residency
        }
    }
}
