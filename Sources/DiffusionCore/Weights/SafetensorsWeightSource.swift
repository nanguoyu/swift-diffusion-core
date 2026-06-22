@preconcurrency import MLX
import Foundation

public enum WeightSourceError: Error, CustomStringConvertible {
    case missingTensor(String)
    case noFiles
    public var description: String {
        switch self {
        case .missingTensor(let k): return "WeightSource: missing tensor '\(k)'"
        case .noFiles: return "WeightSource: no safetensors files provided"
        }
    }
}

/// A `WeightSource` backed by one or more safetensors files loaded via MLX.
///
/// MLX memory-maps the file and creates lazy arrays, so a tensor is materialized only on
/// first use. This is the internal-storage source; the external-SSD ranged-read variant
/// (`RangedFileWeightSource`) lands in Phase 3 behind the same protocol.
public final class SafetensorsWeightSource: WeightSource, @unchecked Sendable {
    private let tensors: [String: MLXArray]
    public let isStreaming: Bool

    public init(tensors: [String: MLXArray], isStreaming: Bool = false) {
        self.tensors = tensors
        self.isStreaming = isStreaming
    }

    /// Merge-load one or more `.safetensors` files (e.g. sharded transformer + encoder + vae).
    public convenience init(files: [URL], isStreaming: Bool = false) throws {
        guard !files.isEmpty else { throw WeightSourceError.noFiles }
        var merged: [String: MLXArray] = [:]
        for url in files {
            for (key, value) in try loadArrays(url: url) { merged[key] = value }
        }
        self.init(tensors: merged, isStreaming: isStreaming)
    }

    public func tensor(_ key: TensorKey) throws -> MLXArray {
        guard let array = tensors[key.name] else { throw WeightSourceError.missingTensor(key.name) }
        return array
    }

    public var tensorNames: [String] { Array(tensors.keys) }
    public var count: Int { tensors.count }
}
