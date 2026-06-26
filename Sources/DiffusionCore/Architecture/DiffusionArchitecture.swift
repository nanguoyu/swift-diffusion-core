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

    /// Build the denoiser (patch-embed → streamable blocks → unembed) for this run.
    func makeDenoiser(source: WeightSource) throws -> any Denoiser

    /// Prepare the initial latent for `size`/`seed` (+ optional img2img reference).
    func initialLatent(size: ImageSize, seed: UInt64, reference: CGImage?, strength: Float,
                       source: WeightSource) throws -> MLXArray

    /// Decode the final latent to an image (VAE).
    func decode(_ latent: MLXArray, source: WeightSource) async throws -> CGImage

    /// Free the text encoder after `encode` returns. The engine calls this once the
    /// conditioning is captured (two-phase staging) so the encoder and transformer never
    /// co-reside. Default is a no-op for architectures that don't hold a releasable encoder.
    func releaseTextEncoder()

    /// Free any architecture-owned caches retained across phases or generations, such as a
    /// lazily loaded decoder. Default is a no-op for stateless architectures.
    func releaseCachedResources()

    /// The sigma schedule for this run (must return `steps + 1` values, trailing 0 included).
    ///
    /// Defaults to the engine's sampler, so an architecture whose schedule that sampler reproduces
    /// (e.g. Z-Image) needs no override and stays byte-for-byte. FLUX overrides this: its schedule
    /// uses an empirical `mu` that depends on BOTH the image sequence length AND the step count, so
    /// no fixed-shift sampler can reproduce it — a streamed FLUX run that used the default sampler
    /// would denoise on a different schedule than the resident pipeline and produce a different image.
    /// The architecture owns the seqLen↔size mapping (it knows its own patch/VAE packing).
    func sigmas(size: ImageSize, steps: Int, sampler: any Sampler) -> [Float]
}

public extension DiffusionArchitecture {
    func releaseTextEncoder() {}
    func releaseCachedResources() {}
    func sigmas(size: ImageSize, steps: Int, sampler: any Sampler) -> [Float] {
        sampler.timesteps(steps: steps)
    }
}

/// The denoiser: patch-embed → N streamable blocks → unembed to a velocity/noise prediction.
/// Splitting embed/unembed from the blocks lets the engine stream the (large) blocks while
/// embed/unembed (small, resident) stay loaded.
public protocol Denoiser: AnyObject {
    /// The ordered transformer blocks, each independently loadable/releasable.
    var blocks: [any StreamableBlock] { get }
    /// Map the latent (+ timestep + conditioning) into the block hidden state.
    func embed(latent: MLXArray, timestep: MLXArray, conditioning: Conditioning) -> MLXArray
    /// Map the final hidden state to the model output (velocity / noise) in *latent* space —
    /// any patching/packing done in `embed` must be fully reversed here so the element-wise
    /// flow-match step stays in latent space.
    func unembed(_ hidden: MLXArray) -> MLXArray
}

/// Static facts the engine needs to drive an architecture.
public struct ArchitectureSpec: Sendable {
    public let family: ModelFamily
    public let latentChannels: Int
    public let defaultSampler: SamplerKind
    public let defaultSteps: Int
    public let defaultGuidance: Float
    /// VAE latent scaling for a cheap linear latent→RGB preview. FLUX.2 uses scale 1; the
    /// SD family uses ≈ 0.18215. `decode`/`initialLatent` apply the real scaling internally.
    public let vaeScale: Float
    public let vaeShift: Float
    /// Rectified-flow sigma-schedule skew (`FlowMatchEulerSampler.shift`). Distilled checkpoints
    /// train with a shifted schedule; the engine builds its sampler from this so an architecture
    /// can never silently run the plain shift = 1 schedule.
    public let samplerShift: Float
    /// Diffusers `shift_terminal` for the sampler (Z-Image = 0.02; 0 disables). Drives the last
    /// working sigma so the distilled model isn't evaluated below its calibrated range.
    public let samplerShiftTerminal: Float

    public init(family: ModelFamily, latentChannels: Int, defaultSampler: SamplerKind,
                defaultSteps: Int, defaultGuidance: Float,
                vaeScale: Float = 1.0, vaeShift: Float = 0.0,
                samplerShift: Float = 1.0, samplerShiftTerminal: Float = 0.0) {
        self.family = family
        self.latentChannels = latentChannels
        self.defaultSampler = defaultSampler
        self.defaultSteps = defaultSteps
        self.defaultGuidance = defaultGuidance
        self.vaeScale = vaeScale
        self.vaeShift = vaeShift
        self.samplerShift = samplerShift
        self.samplerShiftTerminal = samplerShiftTerminal
    }
}

/// Output of the text encoder, consumed by every denoiser block.
public struct Conditioning: Sendable {
    public let embeddings: MLXArray
    public let pooled: MLXArray?
    /// Architecture-specific conditioning tensors threaded into `embed` and every block —
    /// e.g. 2D image position ids / rotary tables for FLUX double-stream and Z-Image attention.
    public let extras: [String: MLXArray]
    public init(embeddings: MLXArray, pooled: MLXArray? = nil, extras: [String: MLXArray] = [:]) {
        self.embeddings = embeddings
        self.pooled = pooled
        self.extras = extras
    }
}
