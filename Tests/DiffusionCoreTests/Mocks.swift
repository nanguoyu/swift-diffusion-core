@preconcurrency import MLX
import CoreGraphics
import Foundation
@testable import DiffusionCore

/// Thread-safe step collector for asserting per-step progress from a `@Sendable` callback.
final class StepRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var _steps: [Int] = []
    func record(_ s: Int) { lock.lock(); _steps.append(s); lock.unlock() }
    var steps: [Int] { lock.lock(); defer { lock.unlock() }; return _steps }
}

/// A trivial denoiser block: keeps shape, exercises the MLX graph, and counts load/release so
/// tests can assert the engine's residency behavior (resident vs. per-step streaming).
final class MockBlock: StreamableBlock {
    let index: Int
    let approximateBytes: Int64 = 1_000_000
    private(set) var loadCount = 0
    private(set) var releaseCount = 0
    init(index: Int) { self.index = index }
    func load(from source: WeightSource) throws { loadCount += 1 }
    func callAsFunction(_ x: MLXArray, conditioning: Conditioning, timestep: MLXArray) -> MLXArray {
        x * 0.99
    }
    func release() { releaseCount += 1 }
}

final class MockDenoiser: Denoiser {
    let mockBlocks: [MockBlock]
    var blocks: [any StreamableBlock] { mockBlocks }
    init(n: Int) { mockBlocks = (0..<n).map { MockBlock(index: $0) } }
    func embed(latent: MLXArray, timestep: MLXArray, conditioning: Conditioning) -> MLXArray { latent }
    func unembed(_ hidden: MLXArray) -> MLXArray { hidden }
}

final class MockArchitecture: DiffusionArchitecture, @unchecked Sendable {
    static let spec = ArchitectureSpec(family: .zImage, latentChannels: 3,
                                       defaultSampler: .flowMatchEuler, defaultSteps: 4, defaultGuidance: 1.0)
    let denoiser: MockDenoiser
    init(blocks: Int) { denoiser = MockDenoiser(n: blocks) }

    func encode(_ prompt: String, negative: String?, source: WeightSource) async throws -> Conditioning {
        Conditioning(embeddings: MLXArray(converting: [0.0], [1, 1]).asType(.float32))
    }
    func initialLatent(size: ImageSize, seed: UInt64, reference: CGImage?, strength: Float,
                       source: WeightSource) throws -> MLXArray {
        let h = size.height / 8, w = size.width / 8
        return MLXArray(converting: [Double](repeating: 0.5, count: h * w * 3), [h, w, 3]).asType(.float32)
    }
    func makeDenoiser(source: WeightSource) throws -> any Denoiser { denoiser }
    func decode(_ latent: MLXArray, source: WeightSource) async throws -> CGImage {
        guard let image = ImageConversion.cgImage(fromHWC: latent) else { throw EngineError.decodeFailed }
        return image
    }
}

struct MockWeightSource: WeightSource {
    let isStreaming: Bool
    let freesOnRelease = true   // mock blocks don't retain source tensors
    func tensor(_ key: TensorKey) throws -> MLXArray { MLXArray(converting: [0.0]) }
}
