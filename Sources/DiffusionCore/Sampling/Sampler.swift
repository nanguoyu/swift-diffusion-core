import MLX

/// Schedulers/samplers the engine can run. Each architecture declares a default; the user
/// may override where it makes sense.
public enum SamplerKind: String, Sendable, CaseIterable {
    case flowMatchEuler
    case eulerAncestral
    case dpmSolverMultistep
    case ddim
}

/// A denoising sampler. The engine owns the step loop and calls `step` after each model pass.
public protocol Sampler: Sendable {
    var kind: SamplerKind { get }

    /// Timesteps for `steps` (and optional shift), high → low noise.
    func timesteps(steps: Int) -> [Float]

    /// Combine the current latent with the model's predicted noise/velocity for one step.
    func step(latent: MLXArray, modelOutput: MLXArray, t: Float, tPrev: Float) -> MLXArray
}
