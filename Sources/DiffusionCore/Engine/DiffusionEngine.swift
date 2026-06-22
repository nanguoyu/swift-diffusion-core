import CoreGraphics

/// The single boundary the app talks to.
///
/// One concrete implementation — `MLXDiffusionEngine` — drives any model by consuming the
/// model package's `DiffusionArchitecture`, applying the streaming/partial-load ladder as
/// the `MemoryGovernor` dictates. The app never imports a specific architecture.
public protocol DiffusionEngine: Sendable {

    /// What a given model can do on a given device, and *how* it will run there
    /// (resident vs. streamed). Drives the gallery's hardware-fit badges.
    static func capabilities(for model: DiffusionModel,
                             variant: ModelVariant,
                             on device: DeviceTier) -> EngineCapabilities

    /// Build the pipeline for `model` reading weights from `source`. Lazy: call before the
    /// first `generate`. `progress` reports 0...1 of model load/prepare.
    func load(_ model: DiffusionModel,
              variant: ModelVariant,
              source: WeightSource,
              progress: @Sendable @escaping (Double) -> Void) async throws

    /// Run one generation. `progress` reports download/build/per-step phases.
    func generate(_ request: GenerationRequest,
                  progress: @Sendable @escaping (GenerationProgress) -> Void) async throws -> CGImage

    /// Free all resident weights (on memory warning, backgrounding, or model switch).
    func unload() async
}
