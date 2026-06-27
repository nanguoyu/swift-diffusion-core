@preconcurrency import MLX
import CoreGraphics

public enum EngineError: Error, CustomStringConvertible {
    case notLoaded
    case decodeFailed
    case invalidRequest(String)
    case streamingUnavailable
    case unsupportedOnDevice
    /// Generation paused because the device got too hot and did not cool within the bounded window.
    /// Recoverable — the caller should surface a "let your phone cool down" message and allow retry,
    /// never treat it as a hard failure.
    case pausedForHeat
    public var description: String {
        switch self {
        case .notLoaded: return "DiffusionEngine: no model loaded"
        case .decodeFailed: return "DiffusionEngine: VAE decode produced no image"
        case .invalidRequest(let m): return "DiffusionEngine: invalid request — \(m)"
        case .streamingUnavailable:
            return "DiffusionEngine: this model needs a streaming weight source, which is not available yet"
        case .unsupportedOnDevice: return "DiffusionEngine: model does not fit this device"
        case .pausedForHeat: return "DiffusionEngine: paused to let the device cool down"
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
    /// Streaming knob: materialize (eval + clearCache) every K blocks instead of every block.
    /// 1 keeps the proven per-block residency plateau; larger lets K block runs pipeline on the
    /// GPU before a sync, trading ~K blocks of peak weight residency for fewer stalls.
    private let streamEvalEveryK: Int
    /// The image token count this engine is being built to render, used to size the activation
    /// working set at load() time — BEFORE the first GenerationRequest exists. The app rebuilds the
    /// engine when the request crosses the 512↔1024 residency band, so the target size is known here.
    /// `nil` ⇒ the reference (512 px) — keeping Z-Image and the macOS/test callers byte-for-byte.
    private let targetImageSeqLen: Int?

    public init(architecture: any DiffusionArchitecture,
                sampler: (any Sampler)? = nil,
                device: DeviceTier = .current,
                streamEvalEveryK: Int = 1,
                targetImageSeqLen: Int? = nil) {
        self.streamEvalEveryK = max(1, streamEvalEveryK)
        self.targetImageSeqLen = targetImageSeqLen
        self.architecture = architecture
        // Build the sampler from the architecture's spec so it gets the right schedule skew (e.g.
        // Z-Image's shift = 3) instead of silently defaulting to the plain shift = 1 schedule.
        self.sampler = sampler ?? FlowMatchEulerSampler(shift: type(of: architecture).spec.samplerShift,
                                                        shiftTerminal: type(of: architecture).spec.samplerShiftTerminal)
        self.device = device
    }

    public static func capabilities(for model: DiffusionModel, variant: ModelVariant,
                                    on device: DeviceTier) -> EngineCapabilities {
        // Protocol requirement: reference-resolution fit (512 px). The app's Z-Image badge calls this
        // (Z-Image renders at the reference seqLen on iPhone), so its number is unchanged.
        capabilities(for: model, variant: variant, on: device, imageSeqLen: nil)
    }

    /// Sequence-aware fit. `imageSeqLen` (image token count = (W/16)·(H/16); 512 px = 1024, 1024 px =
    /// 4096) scales the working set so the badge for a large render reflects the streaming plan it
    /// will actually take. `nil` ⇒ reference (512 px), identical to the protocol method.
    public static func capabilities(for model: DiffusionModel, variant: ModelVariant,
                                    on device: DeviceTier, imageSeqLen: Int?) -> EngineCapabilities {
        // Residency (resident vs block-streaming) is memory-driven for every family — including FLUX.2,
        // which now streams on iPhone instead of being macOS-only. The app routes 512 to the resident
        // facade and 1024 to this streaming engine; the plan here is the fit the gallery badge shows.
        return MemoryGovernor.plan(variant: variant, device: device,
                                   externalSSDAvailable: false, imageSeqLen: imageSeqLen)
    }

    public func load(_ model: DiffusionModel, variant: ModelVariant, source: WeightSource,
                     progress: @Sendable @escaping (Double) -> Void) async throws {
        let plan = MemoryGovernor.plan(variant: variant, device: device,
                                       externalSSDAvailable: source.isStreaming,
                                       imageSeqLen: targetImageSeqLen)
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
        // transformer loads. Awaited so a @MainActor encoder singleton is actually freed before the
        // transformer begins streaming. No-op for architectures without a releasable encoder.
        await architecture.releaseTextEncoder()

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
        // The architecture owns its sigma schedule (default = sampler.timesteps, so Z-Image is
        // unchanged; FLUX overrides because its empirical-mu schedule is seqLen+steps dependent and
        // a fixed-shift sampler cannot reproduce it — parity-critical for the streamed path).
        let sigmas = architecture.sigmas(size: request.size, steps: request.steps, sampler: sampler)
        guard sigmas.count == request.steps + 1 else {
            throw EngineError.invalidRequest("sampler returned \(sigmas.count) sigmas for \(request.steps) steps")
        }
        for i in 0 ..< request.steps {
            try request.control?.checkpoint()
            let t = sigmas[i], tNext = sigmas[i + 1]
            let timestep = MLXArray(t)

            var hidden = denoiser.embed(latent: latent, timestep: timestep, conditioning: conditioning)
            for (blockIdx, block) in denoiser.blocks.enumerated() {
                try request.control?.checkpoint()
                if streaming { try block.load(from: source) }
                hidden = block(hidden, conditioning: conditioning, timestep: timestep)
                if streaming {
                    block.release()
                    // Coarse materialization: sync every K blocks (default 1) so runs pipeline; K
                    // blocks' weights stay live until the eval, so peak grows ~K blocks. Bit-exact —
                    // eval cadence changes nothing about the values.
                    let isLastBlock = blockIdx == denoiser.blocks.count - 1
                    if (blockIdx + 1) % streamEvalEveryK == 0 || isLastBlock {
                        eval(hidden)
                        MLX.GPU.clearCache()
                    }
                }
            }
            let velocity = denoiser.unembed(hidden)

            latent = sampler.step(latent: latent, modelOutput: velocity, t: t, tPrev: tNext)
            eval(latent)
            progress(.denoising(step: i + 1, total: request.steps, preview: nil))
            // Thermal pacing between steps: a no-op on macOS and when cool; inserts cooperative
            // sleeps / a cooling pause on a hot phone so a long run slows or pauses rather than
            // tripping an OS thermal shutdown. Cancellation flows through Task.sleep.
            try await ThermalGovernor.shared.throttleIfNeeded { progress(.cooling) }
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
