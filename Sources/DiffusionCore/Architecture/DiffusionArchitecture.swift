@preconcurrency import MLX
import CoreGraphics

/// The seam every model package implements. The core engine drives this without knowing the
/// specific architecture (Z-Image single-stream S3-DiT, FLUX double-stream, MMDiT, …), so
/// adding a model = adding a package that conforms here.
public protocol DiffusionArchitecture: Sendable {

    static var spec: ArchitectureSpec { get }

    /// Encode the prompt. The text encoder is loaded for this call; the engine releases it
    /// afterwards (two-phase staging).
    func encode(_ prompt: String, negative: String?, source: WeightSource) async throws -> Conditioning

    /// The denoiser as an ordered list of independently loadable blocks (for streaming).
    func denoiserBlocks(source: WeightSource) throws -> [any StreamableBlock]

    /// Prepare the initial latent for `size`/`seed` (+ optional img2img reference).
    func initialLatent(size: ImageSize, seed: UInt64, reference: CGImage?, strength: Float,
                       source: WeightSource) throws -> MLXArray

    /// Decode the final latent to an image (VAE).
    func decode(_ latent: MLXArray, source: WeightSource) async throws -> CGImage
}

/// Static facts the engine needs to drive an architecture.
public struct ArchitectureSpec: Sendable {
    public let family: ModelFamily
    public let latentChannels: Int
    public let defaultSampler: SamplerKind
    public let defaultSteps: Int
    public let defaultGuidance: Float

    public init(family: ModelFamily, latentChannels: Int, defaultSampler: SamplerKind,
                defaultSteps: Int, defaultGuidance: Float) {
        self.family = family
        self.latentChannels = latentChannels
        self.defaultSampler = defaultSampler
        self.defaultSteps = defaultSteps
        self.defaultGuidance = defaultGuidance
    }
}

/// Output of the text encoder, consumed by every denoiser block.
public struct Conditioning: Sendable {
    public let embeddings: MLXArray
    public let pooled: MLXArray?
    public init(embeddings: MLXArray, pooled: MLXArray? = nil) {
        self.embeddings = embeddings
        self.pooled = pooled
    }
}
