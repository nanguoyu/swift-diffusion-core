@preconcurrency import MLX
import Foundation

public enum WeightSourceError: Error, CustomStringConvertible {
    case missingTensor(String)
    case noFiles
    case duplicateTensor(String)
    public var description: String {
        switch self {
        case .missingTensor(let k): return "WeightSource: missing tensor '\(k)'"
        case .noFiles: return "WeightSource: no safetensors files provided"
        case .duplicateTensor(let k): return "WeightSource: duplicate tensor key '\(k)' across files"
        }
    }
}

/// A `WeightSource` backed by one or more safetensors files loaded via MLX.
///
/// MLX memory-maps the file and creates lazy arrays, so a tensor is materialized only on
/// first use. This is the internal-storage source; the external-SSD ranged-read variant
/// (`RangedFileWeightSource`) lands in Phase 3 behind the same protocol.
// `@unchecked Sendable` is safe here: storage is an immutable `let` dictionary and the
// returned `MLXArray`s must be treated as read-only by callers.
public final class SafetensorsWeightSource: WeightSource, @unchecked Sendable {
    private let tensors: [String: MLXArray]
    public let isStreaming: Bool

    /// This source holds every tensor resident for its whole lifetime, so dropping a block's
    /// reference does not free memory. It is a *resident* source; a true streaming source
    /// (`RangedFileWeightSource`, Phase 3) will read tensors on demand and set this `true`.
    public let freesOnRelease = false

    public init(tensors: [String: MLXArray], isStreaming: Bool = false) {
        self.tensors = tensors
        self.isStreaming = isStreaming
    }

    /// Merge-load one or more `.safetensors` files (e.g. sharded transformer + encoder + vae).
    /// Throws on duplicate tensor keys across files so heterogeneous components can't silently
    /// cross-wire on generic names like `norm.weight`.
    public convenience init(files: [URL], isStreaming: Bool = false) throws {
        guard !files.isEmpty else { throw WeightSourceError.noFiles }
        var merged: [String: MLXArray] = [:]
        for url in files {
            for (key, value) in try loadArrays(url: url) {
                guard merged[key] == nil else { throw WeightSourceError.duplicateTensor(key) }
                merged[key] = value
            }
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
