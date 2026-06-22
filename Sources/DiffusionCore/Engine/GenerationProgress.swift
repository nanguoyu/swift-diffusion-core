import CoreGraphics

/// Progress emitted during a generation, spanning every user-visible phase.
public enum GenerationProgress: Sendable {
    /// Resolving / downloading missing weights (0...1).
    case downloading(fraction: Double)
    /// Building / compiling the pipeline ("preparing model…").
    case preparing
    /// Encoding the prompt (text encoder resident).
    case encoding
    /// Denoising. `step`/`total`; optional cheap latent preview.
    case denoising(step: Int, total: Int, preview: CGImage?)
    /// Decoding the final latent (VAE).
    case decoding
    /// Done.
    case finished(CGImage)
}
