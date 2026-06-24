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
    /// Diffusers `shift_terminal`: stretch the schedule so the last WORKING sigma lands exactly here
    /// (then a 0 is appended). Distilled checkpoints (Z-Image = 0.02) are only calibrated down to this
    /// sigma — running the final step from far below it (e.g. 0.003) yields a noisy velocity that
    /// surfaces as grain. `0` disables the stretch.
    public let shiftTerminal: Float
    public let numTrainTimesteps: Int

    public init(shift: Float = 1.0, shiftTerminal: Float = 0.0, numTrainTimesteps: Int = 1000) {
        self.shift = shift
        self.shiftTerminal = shiftTerminal
        self.numTrainTimesteps = numTrainTimesteps
    }

    /// `steps + 1` sigmas for the rectified-flow Euler integration. Matches diffusers'
    /// `FlowMatchEulerDiscreteScheduler` (as mflux drives Z-Image): the `steps` working sigmas are
    /// `linspace(1, 1/numTrainTimesteps, steps)`, skewed by `shift`, optionally stretched so the last
    /// lands on `shiftTerminal`, then a SEPARATE trailing 0 is appended. The exponential time-shift
    /// `exp(mu)/(exp(mu)+(1/s−1))` with `exp(mu)=shift` is algebraically identical to this linear form.
    public func timesteps(steps: Int) -> [Float] {
        precondition(steps > 0, "steps must be positive")
        let sigmaMin = 1.0 / Float(numTrainTimesteps)
        // Pre-shift sigmas: linspace(1, sigmaMin, steps) — exactly `steps` points, ending at sigmaMin.
        let pre: [Float] = steps == 1
            ? [1.0]
            : (0 ..< steps).map { 1.0 - (1.0 - sigmaMin) * Float($0) / Float(steps - 1) }
        var sig = shift == 1.0 ? pre : pre.map { (shift * $0) / (1 + (shift - 1) * $0) }
        if shiftTerminal > 0, let last = sig.last, last < 1 {
            let scale = (1 - last) / (1 - shiftTerminal)
            sig = sig.map { 1 - (1 - $0) / scale }
        }
        return sig + [0.0]
    }

    /// One Euler step along the flow: `x_next = x + (sigma_next - sigma) * v`.
    public func step(latent: MLXArray, modelOutput: MLXArray, t: Float, tPrev: Float) -> MLXArray {
        latent + modelOutput * (tPrev - t)
    }
}
