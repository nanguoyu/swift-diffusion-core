import Foundation

/// The built-in model catalog. Static order == gallery order. Sizes are measured on-disk
/// values; the exact file list per repo is resolved at download time. Public, openly-licensed
/// models only.
public enum ModelCatalog {

    public static let all: [DiffusionModel] = [zImageTurbo, fluxKlein4B]

    private static func gb(_ x: Double) -> Int64 { Int64(x * 1_000_000_000) }

    // MARK: Z-Image Turbo — Tongyi, 6B, single-stream S3-DiT, Qwen3-4B text encoder, 8-step.
    public static let zImageTurbo = DiffusionModel(
        id: "z-image-turbo",
        displayName: "Z-Image Turbo",
        family: .zImage,
        publisher: "Tongyi",
        summary: "6B single-stream diffusion transformer, distilled for 8-step, sub-second, bilingual generation.",
        license: .apache2,
        architecture: ArchitectureSpec(family: .zImage, latentChannels: 16,
                                        defaultSampler: .flowMatchEuler, defaultSteps: 8, defaultGuidance: 1.0),
        variants: [
            ModelVariant(precision: .q8, approximateBytes: gb(11.0),
                         components: ComponentSizes(transformer: gb(6.54), textEncoder: gb(4.27), vae: gb(0.17)),
                         layout: .mfluxShard,
                         source: ModelSource(huggingFaceRepo: "deepsweet/Z-Image-Turbo-6B-MLX-Q8")),
            ModelVariant(precision: .q4, approximateBytes: gb(5.9),
                         components: ComponentSizes(transformer: gb(3.46), textEncoder: gb(2.26), vae: gb(0.16)),
                         layout: .mfluxShard,
                         source: ModelSource(huggingFaceRepo: "deepsweet/Z-Image-Turbo-6B-MLX-Q4")),
        ])

    // MARK: FLUX.2 Klein 4B — Black Forest, DiT. Uses the existing flux-2-swift-mlx package.
    public static let fluxKlein4B = DiffusionModel(
        id: "flux2-klein-4b",
        displayName: "FLUX.2 Klein 4B",
        family: .flux2,
        publisher: "Black Forest",
        summary: "Distilled FLUX.2 DiT — the lightest model; runs on Mac via the FLUX facade engine.",
        license: .apache2,
        architecture: ArchitectureSpec(family: .flux2, latentChannels: 128,
                                        defaultSampler: .flowMatchEuler, defaultSteps: 6, defaultGuidance: 1.0),
        variants: [
            ModelVariant(precision: .q8, approximateBytes: gb(8.57),
                         components: ComponentSizes(transformer: gb(4.12), textEncoder: gb(4.27), vae: gb(0.17)),
                         layout: .mfluxShard,
                         source: ModelSource(huggingFaceRepo: "mlx-community/flux2-klein-4b-8bit")),
            ModelVariant(precision: .q4, approximateBytes: gb(4.62),
                         components: ComponentSizes(transformer: gb(2.18), textEncoder: gb(2.26), vae: gb(0.17)),
                         layout: .mfluxShard,
                         source: ModelSource(huggingFaceRepo: "mlx-community/flux2-klein-4b-4bit")),
        ])
}
