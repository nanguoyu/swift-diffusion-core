import MLX

/// Identifies a single weight tensor within a model's on-disk files.
public struct TensorKey: Hashable, Sendable {
    public let name: String
    public init(_ name: String) { self.name = name }
}

/// Abstracts *where* and *how* a tensor's bytes are read, so the same engine runs from
/// internal storage or streams from an external USB-C SSD. Reads are by tensor (range),
/// not bound to mmap — the key to making external-SSD streaming a first-class source.
///
/// Planned implementations:
///   - `MmapWeightSource`        — mmap the safetensors file (internal, fastest)
///   - `RangedFileWeightSource`  — `pread` byte ranges on demand (external SSD)
///   - `HybridWeightSource`      — hot tensors resident, cold tensors streamed + prefetched
public protocol WeightSource: Sendable {
    /// Materialize a single tensor as an `MLXArray`. May read lazily / on demand.
    func tensor(_ key: TensorKey) throws -> MLXArray

    /// True if backed by slow/removable storage (external SSD) — the governor uses this to
    /// pick streaming vs. resident and to warn "keep the drive connected".
    var isStreaming: Bool { get }

    /// True if dropping a tensor reference actually frees its buffer (an on-demand / evicting
    /// source). The engine refuses a streaming residency plan unless this is true — otherwise
    /// `StreamableBlock.release()` saves nothing. Fully-resident sources return `false`.
    var freesOnRelease: Bool { get }
}

public extension WeightSource {
    var freesOnRelease: Bool { false }
}
