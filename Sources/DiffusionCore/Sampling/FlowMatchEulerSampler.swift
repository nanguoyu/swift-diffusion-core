@preconcurrency import MLX

/// Flow-matching Euler sampler (rectified flow).
///
/// Integrates the model's velocity field from noise (sigma = 1) to data (sigma = 0). This is
/// the default scheduler for FLUX.2 and Z-Image. `shift` skews the sigma schedule toward
/// higher noise (matching the distilled checkpoints' training); `shift = 1` is the plain
/// linear schedule.
public struct FlowMatchEulerSampler: Sampler {
    public let kind: SamplerKind = .flowMatchEuler
    public let shift: Float

    public init(shift: Float = 1.0) { self.shift = shift }

    /// `steps + 1` sigmas, linearly spaced 1 → 0, optionally shifted. The last is 0 (clean).
    public func timesteps(steps: Int) -> [Float] {
        precondition(steps > 0, "steps must be positive")
        let linear = (0...steps).map { 1.0 - Float($0) / Float(steps) }
        guard shift != 1.0 else { return linear }
        return linear.map { s in s <= 0 ? 0 : (shift * s) / (1 + (shift - 1) * s) }
    }

    /// One Euler step along the flow: `x_next = x + (sigma_next - sigma) * v`.
    public func step(latent: MLXArray, modelOutput: MLXArray, t: Float, tPrev: Float) -> MLXArray {
        latent + modelOutput * (tPrev - t)
    }
}
