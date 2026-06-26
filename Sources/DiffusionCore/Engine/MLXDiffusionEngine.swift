@preconcurrency import MLX
import CoreGraphics

public enum EngineError: Error, CustomStringConvertible {
    case notLoaded
    case decodeFailed
    case invalidRequest(String)
    case streamingUnavailable
    case unsupportedOnDevice
    public var description: String {
        switch self {
        case .notLoaded: return "DiffusionEngine: no model loaded"
        case .decodeFailed: return "DiffusionEngine: VAE decode produced no image"
        case .invalidRequest(let m): return "DiffusionEngine: invalid request — \(m)"
        case .streamingUnavailable:
            return "DiffusionEngine: this model needs a streaming weight source, which is not available yet"
        case .unsupportedOnDevice: return "DiffusionEngine: model does not fit this device"
        }
    }
}

/// The generic MLX engine. An `actor` so concurrent load/generate/unload can't race the
/// residency state. It drives any `DiffusionArchitecture` and applies the partial-load ladder
/// the `MemoryGovernor` selects: when streaming, each denoiser block is loaded, run, released,
/// and the MLX reuse pool is cleared per block so peak weight residency actually plateaus;
/// otherwise all blocks stay resident for speed.
///
/// FLUX.2 (via flux-2-swift-mlx) is monolithic and macOS-only, so it is wrapped by a separate
/// facade engine — not by this block-streaming path. This path powers Z-Image and the iPhone.
public actor MLXDiffusionEngine: DiffusionEngine {
    private let architecture: any DiffusionArchitecture
    private let sampler: any Sampler
    private let device: DeviceTier
    private var source: WeightSource?
    private var residency: EngineCapabilities.Residency = .resident

    public init(architecture: any DiffusionArchitecture,
                sampler: (any Sampler)? = nil,
                device: DeviceTier = .current) {
        self.architecture = architecture
        // Build the sampler from the architecture's spec so it gets the right schedule skew (e.g.
        // Z-Image's shift = 3) instead of silently defaulting to the plain shift = 1 schedule.
        self.sampler = sampler ?? FlowMatchEulerSampler(shift: type(of: architecture).spec.samplerShift,
                                                        shiftTerminal: type(of: architecture).spec.samplerShiftTerminal)
        self.device = device
    }

    public static func capabilities(for model: DiffusionModel, variant: ModelVariant,
                                    on device: DeviceTier) -> EngineCapabilities {
        // FLUX.2 is macOS-only (handled by the facade engine), so it never "runs" on a phone here.
        if model.family == .flux2 && device.isPhone {
            return EngineCapabilities(runnable: false, residency: .unsupported,
                                      estimatedPeakBytes: variant.approximateBytes, note: "macOS only")
        }
        return MemoryGovernor.plan(variant: variant, device: device, externalSSDAvailable: false)
    }

    public func load(_ model: DiffusionModel, variant: ModelVariant, source: WeightSource,
                     progress: @Sendable @escaping (Double) -> Void) async throws {
        let plan = MemoryGovernor.plan(variant: variant, device: device,
                                       externalSSDAvailable: source.isStreaming)
        guard plan.runnable else { throw EngineError.unsupportedOnDevice }

        // Step the residency rung down if live process headroom is below the planned peak.
        var residency = plan.residency
        if MemoryGovernor.availableBytesNow() < plan.estimatedPeakBytes {
            residency = MemoryGovernor.leaner(than: residency)
        }

        let streaming = residency == .streamingInternal || residency == .streamingExternal
        // A streaming plan only saves memory if the source actually frees on release.
        if streaming, !source.freesOnRelease { throw EngineError.streamingUnavailable }

        // Bound the MLX reuse pool so weights freed by release() can actually plateau.
        let cacheLimit = streaming ? 384_000_000
                                   : max(256_000_000, Int(Double(device.memoryBudgetBytes) * 0.5))
        MLX.GPU.set(cacheLimit: cacheLimit)

        self.source = source
        self.residency = residency
        progress(1.0)
    }

    public func generate(_ request: GenerationRequest,
                         progress: @Sendable @escaping (GenerationProgress) -> Void) async throws -> CGImage {
        guard request.steps > 0 else {
            throw EngineError.invalidRequest("steps must be positive, got \(request.steps)")
        }
        guard let source else { throw EngineError.notLoaded }
        let streaming = residency == .streamingInternal || residency == .streamingExternal
        try request.control?.checkpoint()

        progress(.encoding)
        let conditioning = try await architecture.encode(request.prompt,
                                                         negative: request.negativePrompt,
                                                         source: source)
        try request.control?.checkpoint()
        // Two-phase staging: the encoder output is captured, so free the encoder before the
        // transformer loads. No-op for architectures without a releasable encoder.
        architecture.releaseTextEncoder()

        var latent = try architecture.initialLatent(size: request.size, seed: request.seed,
                                                     reference: request.referenceImage,
                                                     strength: request.strength, source: source)
        try request.control?.checkpoint()

        let denoiser = try architecture.makeDenoiser(source: source)
        if !streaming { for block in denoiser.blocks { try block.load(from: source) } }
        defer {
            if !streaming { for block in denoiser.blocks { block.release() } }
            MLX.GPU.clearCache()
        }

        progress(.preparing)
        let sigmas = sampler.timesteps(steps: request.steps)
        guard sigmas.count == request.steps + 1 else {
            throw EngineError.invalidRequest("sampler returned \(sigmas.count) sigmas for \(request.steps) steps")
        }
        for i in 0 ..< request.steps {
            try request.control?.checkpoint()
            let t = sigmas[i], tNext = sigmas[i + 1]
            let timestep = MLXArray(t)

            var hidden = denoiser.embed(latent: latent, timestep: timestep, conditioning: conditioning)
            for block in denoiser.blocks {
                try request.control?.checkpoint()
                if streaming { try block.load(from: source) }
                hidden = block(hidden, conditioning: conditioning, timestep: timestep)
                if streaming {
                    eval(hidden)
                    block.release()
                    MLX.GPU.clearCache()
                }
            }
            let velocity = denoiser.unembed(hidden)

            latent = sampler.step(latent: latent, modelOutput: velocity, t: t, tPrev: tNext)
            eval(latent)
            progress(.denoising(step: i + 1, total: request.steps, preview: nil))
            try request.control?.checkpoint()
        }
        try request.control?.checkpoint()
        progress(.decoding)
        let image = try await architecture.decode(latent, source: source)
        try request.control?.checkpoint()
        progress(.finished(image))
        return image
    }

    public func unload() async {
        source = nil
        architecture.releaseCachedResources()
        MLX.GPU.clearCache()
    }
}
