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
    public let numTrainTimesteps: Int

    public init(shift: Float = 1.0, numTrainTimesteps: Int = 1000) {
        self.shift = shift
        self.numTrainTimesteps = numTrainTimesteps
    }

    /// `steps + 1` sigmas for the rectified-flow Euler integration. Matches diffusers'
    /// `FlowMatchEulerDiscreteScheduler`: the `steps` working sigmas are spaced from sigma_max = 1
    /// down to sigma_min = `1/numTrainTimesteps` (NOT to 0), optionally skewed by `shift`, and a
    /// SEPARATE trailing 0 (clean) is appended. Folding 0 into the pre-shift grid (the old bug)
    /// warped every intermediate sigma and made the final Euler step oversized (e.g. shift 3 / 8
    /// steps ended …0.5, 0.3, 0 instead of …0.335, 0.003, 0) — which an 8-step distilled velocity
    /// field, calibrated only at its training sigmas, renders as a soft, low-detail image.
    public func timesteps(steps: Int) -> [Float] {
        precondition(steps > 0, "steps must be positive")
        let sigmaMin = 1.0 / Float(numTrainTimesteps)
        // Pre-shift sigmas: linspace(1, sigmaMin, steps) — exactly `steps` points, ending at sigmaMin.
        let pre: [Float] = steps == 1
            ? [1.0]
            : (0 ..< steps).map { 1.0 - (1.0 - sigmaMin) * Float($0) / Float(steps - 1) }
        let shifted = shift == 1.0 ? pre : pre.map { (shift * $0) / (1 + (shift - 1) * $0) }
        return shifted + [0.0]
    }

    /// One Euler step along the flow: `x_next = x + (sigma_next - sigma) * v`.
    public func step(latent: MLXArray, modelOutput: MLXArray, t: Float, tPrev: Float) -> MLXArray {
        latent + modelOutput * (tPrev - t)
    }
}
