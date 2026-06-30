import XCTest
@preconcurrency import MLX
@testable import DiffusionCore

final class EngineTests: XCTestCase {
    private let macTier = DeviceTier(physicalMemoryBytes: 18_000_000_000, isPhone: false)

    func testGenerateRunsFullPipelineResident() async throws {
        let arch = MockArchitecture(blocks: 3)
        let engine = MLXDiffusionEngine(architecture: arch, device: macTier)
        let model = ModelCatalog.zImageTurbo
        let variant = try XCTUnwrap(model.variants.first { $0.precision == .q4 })

        try await engine.load(model, variant: variant,
                              source: MockWeightSource(isStreaming: false), progress: { _ in })

        let recorder = StepRecorder()
        let image = try await engine.generate(
            GenerationRequest(prompt: "an otter", steps: 4, seed: 1, size: .square512),
            progress: { p in if case let .denoising(s, _, _) = p { recorder.record(s) } })

        XCTAssertEqual(image.width, 64)   // 512 / 8
        XCTAssertEqual(image.height, 64)
        XCTAssertEqual(recorder.steps, [1, 2, 3, 4])
        // resident: each block loaded exactly once (not per step)
        XCTAssertEqual(arch.denoiser.mockBlocks.map(\.loadCount), [1, 1, 1])
        XCTAssertEqual(arch.denoiser.mockBlocks.map(\.releaseCount), [1, 1, 1])
    }

    func testStreamingLoadsAndReleasesBlocksPerStep() async throws {
        let phone = DeviceTier(physicalMemoryBytes: 8_000_000_000, isPhone: true)
        // big transformer: two-phase peak exceeds the 8 GB phone budget, but it streams under it
        let comps = ComponentSizes(transformer: 5_000_000_000, textEncoder: 2_300_000_000, vae: 160_000_000)
        let variant = ModelVariant(precision: .q4, approximateBytes: 7_460_000_000, components: comps,
                                   layout: .mfluxShard, source: ModelSource(huggingFaceRepo: "example/streamy"))
        XCTAssertEqual(MemoryGovernor.plan(variant: variant, device: phone, externalSSDAvailable: false).residency,
                       .streamingInternal)

        let arch = MockArchitecture(blocks: 2)
        let engine = MLXDiffusionEngine(architecture: arch, device: phone)
        try await engine.load(ModelCatalog.zImageTurbo, variant: variant,
                              source: MockWeightSource(isStreaming: false), progress: { _ in })

        _ = try await engine.generate(
            GenerationRequest(prompt: "x", steps: 3, seed: 7, size: .square512), progress: { _ in })

        // streaming: each block loaded & released once per step (3 steps)
        XCTAssertEqual(arch.denoiser.mockBlocks.map(\.loadCount), [3, 3])
        XCTAssertEqual(arch.denoiser.mockBlocks.map(\.releaseCount), [3, 3])
    }

    private func streamyVariant() -> ModelVariant {
        let comps = ComponentSizes(transformer: 5_000_000_000, textEncoder: 2_300_000_000, vae: 160_000_000)
        return ModelVariant(precision: .q4, approximateBytes: 7_460_000_000, components: comps,
                            layout: .mfluxShard, source: ModelSource(huggingFaceRepo: "example/streamy"))
    }

    func testStreamingPeakBytesIsTheLeanestPeak() {
        let phone = DeviceTier(physicalMemoryBytes: 8_000_000_000, isPhone: true)
        let variant = streamyVariant()
        let plan = MemoryGovernor.plan(variant: variant, device: phone, externalSSDAvailable: false)
        XCTAssertEqual(plan.residency, .streamingInternal)
        // It IS the peak the streaming plan reports, and well below the resident two-phase peak.
        XCTAssertEqual(MemoryGovernor.streamingPeakBytes(variant: variant), plan.estimatedPeakBytes)
        XCTAssertLessThan(MemoryGovernor.streamingPeakBytes(variant: variant),
                          max(variant.components.textEncoder, variant.components.transformer + variant.components.vae))
    }

    func testLoadRefusesWhenLiveHeadroomBelowStreamingPeak() async throws {
        let phone = DeviceTier(physicalMemoryBytes: 8_000_000_000, isPhone: true)
        let variant = streamyVariant()
        // Live headroom one byte below the leanest peak → starting would jetsam, so load must refuse.
        MemoryGovernor.availableBytesOverride = MemoryGovernor.streamingPeakBytes(variant: variant) - 1
        defer { MemoryGovernor.availableBytesOverride = nil }
        let engine = MLXDiffusionEngine(architecture: MockArchitecture(blocks: 2), device: phone)
        do {
            try await engine.load(ModelCatalog.zImageTurbo, variant: variant,
                                  source: MockWeightSource(isStreaming: false), progress: { _ in })
            XCTFail("expected insufficientMemory when live headroom is below the streaming peak")
        } catch EngineError.insufficientMemory {
            // expected — a recoverable refusal instead of a jetsam crash mid-run
        }
    }

    func testGenerateWithoutLoadThrows() async {
        let engine = MLXDiffusionEngine(architecture: MockArchitecture(blocks: 1), device: macTier)
        do {
            _ = try await engine.generate(GenerationRequest(prompt: "x", steps: 1, seed: 1), progress: { _ in })
            XCTFail("expected notLoaded error")
        } catch {
            XCTAssertTrue(error is EngineError)
        }
    }

    func testStepsZeroThrowsBeforeAnyWork() async {
        // The steps guard is the first statement in generate(), before the source check and any
        // MLX op — so unvalidated UI input fails cleanly instead of tripping a sampler precondition.
        let engine = MLXDiffusionEngine(architecture: MockArchitecture(blocks: 1), device: macTier)
        do {
            _ = try await engine.generate(GenerationRequest(prompt: "x", steps: 0, seed: 1), progress: { _ in })
            XCTFail("expected invalidRequest error")
        } catch let error as EngineError {
            if case .invalidRequest = error {} else { XCTFail("wrong error: \(error)") }
        } catch {
            XCTFail("wrong error type: \(error)")
        }
    }
}
