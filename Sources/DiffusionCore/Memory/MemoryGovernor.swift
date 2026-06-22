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
        return EngineCapabilities(runnable: false, residency: .unsupported,
                                  estimatedPeakBytes: twoPhasePeak, note: "Needs more memory")
    }

    /// Live headroom for this process (bytes). Use this — not nominal RAM — at load time.
    public static func availableBytesNow() -> Int64 { Int64(MemoryProbe.availableBytes()) }
}
