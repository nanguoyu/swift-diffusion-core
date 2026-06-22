@preconcurrency import MLX
import CoreGraphics

public enum EngineError: Error, CustomStringConvertible {
    case notLoaded
    case decodeFailed
    public var description: String {
        switch self {
        case .notLoaded: return "DiffusionEngine: no model loaded"
        case .decodeFailed: return "DiffusionEngine: VAE decode produced no image"
        }
    }
}

/// The generic MLX engine. It drives any `DiffusionArchitecture` and applies the partial-load
/// ladder the `MemoryGovernor` selects: when streaming, each denoiser block is loaded, run,
/// and released per step so peak weight residency stays bounded; otherwise all blocks stay
/// resident for speed.
///
/// FLUX.2 (via flux-2-swift-mlx) is monolithic and macOS-only, so it is wrapped by a separate
/// facade engine — not by this block-streaming path. This path powers Z-Image and the iPhone.
public final class MLXDiffusionEngine: DiffusionEngine, @unchecked Sendable {
    private let architecture: any DiffusionArchitecture
    private let sampler: any Sampler
    private let device: DeviceTier
    private var source: WeightSource?
    private var residency: EngineCapabilities.Residency = .resident

    public init(architecture: any DiffusionArchitecture,
                sampler: any Sampler = FlowMatchEulerSampler(),
                device: DeviceTier = .current) {
        self.architecture = architecture
        self.sampler = sampler
        self.device = device
    }

    public static func capabilities(for model: DiffusionModel, variant: ModelVariant,
                                    on device: DeviceTier) -> EngineCapabilities {
        MemoryGovernor.plan(variant: variant, device: device, externalSSDAvailable: false)
    }

    public func load(_ model: DiffusionModel, variant: ModelVariant, source: WeightSource,
                     progress: @Sendable @escaping (Double) -> Void) async throws {
        self.source = source
        self.residency = MemoryGovernor.plan(variant: variant, device: device,
                                             externalSSDAvailable: source.isStreaming).residency
        progress(1.0)
    }

    private var isStreaming: Bool {
        residency == .streamingInternal || residency == .streamingExternal
    }

    public func generate(_ request: GenerationRequest,
                         progress: @Sendable @escaping (GenerationProgress) -> Void) async throws -> CGImage {
        guard let source else { throw EngineError.notLoaded }

        progress(.encoding)
        let conditioning = try await architecture.encode(request.prompt,
                                                         negative: request.negativePrompt,
                                                         source: source)

        var latent = try architecture.initialLatent(size: request.size, seed: request.seed,
                                                     reference: request.referenceImage,
                                                     strength: request.strength, source: source)

        let denoiser = try architecture.makeDenoiser(source: source)
        if !isStreaming { for block in denoiser.blocks { try block.load(from: source) } }

        progress(.preparing)
        let sigmas = sampler.timesteps(steps: request.steps)
        for i in 0 ..< request.steps {
            let t = sigmas[i], tNext = sigmas[i + 1]
            let timestep = MLXArray(t)

            var hidden = denoiser.embed(latent: latent, timestep: timestep, conditioning: conditioning)
            for block in denoiser.blocks {
                if isStreaming { try block.load(from: source) }
                hidden = block(hidden, conditioning: conditioning, timestep: timestep)
                if isStreaming { eval(hidden); block.release() }
            }
            let velocity = denoiser.unembed(hidden)

            latent = sampler.step(latent: latent, modelOutput: velocity, t: t, tPrev: tNext)
            eval(latent)
            progress(.denoising(step: i + 1, total: request.steps, preview: nil))
        }
        if !isStreaming { for block in denoiser.blocks { block.release() } }

        progress(.decoding)
        let image = try await architecture.decode(latent, source: source)
        progress(.finished(image))
        return image
    }

    public func unload() async { source = nil }
}
