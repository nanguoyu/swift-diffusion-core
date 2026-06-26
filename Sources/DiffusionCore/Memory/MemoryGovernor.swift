import Foundation

/// Decides *how* a model+variant runs on a device — the rung of the partial-load ladder —
/// and produces the `EngineCapabilities` the gallery shows and the engine obeys.
public enum MemoryGovernor {

    /// Rough activation/latent/attention/VAE-decode working set on top of resident weights, at the
    /// REFERENCE resolution (512 px → `referenceImageSeqLen` tokens). Measured on-device the transient
    /// peak above resident weights runs ~1 GB (attention + VAE decode + the encoder's deferred free),
    /// so a sub-GB estimate makes resident plans look cheaper than they run.
    private static let baseWorkingSet: Int64 = 1_000_000_000

    /// Image token count at the reference resolution. FLUX-style packing gives (W/16)·(H/16) tokens,
    /// so 512 px → 32·32 = 1024 tokens. The base working set is calibrated to this point.
    public static let referenceImageSeqLen: Int = 1024

    /// Activation working set scaled to an image token count. The transient above resident weights
    /// is dominated by per-token activations (latents, hidden states, MLP) which grow ~linearly with
    /// the token count; MLX's memory-efficient attention does NOT materialize the full [seq,seq] score
    /// matrix for every head at once, so the realistic activation peak is linear-in-area, not the naive
    /// O(seq²). We therefore scale linearly with seqLen above the reference and never *below* the base
    /// (so ≤512 plans are byte-for-byte identical to before). Calibrated to the two on-device anchors:
    /// ~1 GB transient at 512 px (1024 tokens, resident OK) and a peak that no longer fits resident at
    /// 1024 px (4096 tokens, observed FLUX OOM with the 2.18 GB transformer resident → 4× ≈ 4 GB).
    static func workingSet(forImageSeqLen seqLen: Int) -> Int64 {
        let multiplier = max(1.0, Double(seqLen) / Double(referenceImageSeqLen))
        return Int64(Double(baseWorkingSet) * multiplier)
    }

    /// Plan residency for a variant. `externalSSDAvailable` allows the streaming-external rung
    /// when the model is otherwise too big for memory. `imageSeqLen` (image token count) scales the
    /// activation working set for larger resolutions; `nil` uses the reference (512 px) — keeping
    /// every existing caller's plan identical to before this became sequence-aware.
    public static func plan(variant: ModelVariant, device: DeviceTier,
                            externalSSDAvailable: Bool,
                            imageSeqLen: Int? = nil) -> EngineCapabilities {
        let c = variant.components
        let budget = device.memoryBudgetBytes
        let workingSet = workingSet(forImageSeqLen: imageSeqLen ?? referenceImageSeqLen)

        // Two-phase peak: encoder phase vs. transformer+VAE phase never co-reside.
        let twoPhasePeak = max(c.textEncoder, c.transformer + c.vae) + workingSet
        // Streaming peak: only the encoder (run once) + a few resident blocks + working set.
        let streamingPeak = max(c.textEncoder, 1_200_000_000) + workingSet

        if twoPhasePeak <= Int64(Double(budget) * 0.9) {
            return EngineCapabilities(runnable: true, residency: .resident,
                                      estimatedPeakBytes: twoPhasePeak, note: "Runs great")
        }
        // Two-phase keeps the FULL transformer resident — exactly what block-streaming exists to
        // avoid. Never pick it when its peak exceeds the per-app budget (jetsam is unforgiving on
        // phones); a large transformer that doesn't fit here falls through to streamingInternal.
        if twoPhasePeak <= budget {
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
