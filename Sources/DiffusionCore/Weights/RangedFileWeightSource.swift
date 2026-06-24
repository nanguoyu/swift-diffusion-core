@preconcurrency import MLX
import Foundation

/// A streaming `WeightSource` that reads each tensor's exact byte range from disk on demand
/// (`pread`) and frees its buffer when the returned `MLXArray` is released. This is the source
/// that makes block streaming actually save memory: it reports `freesOnRelease == true`, so the
/// engine will load one denoiser block, run it, drop it, and reclaim — keeping transformer
/// residency at one block instead of the whole model.
///
/// Backed by POSIX `pread` on a read-only fd, so reads are random-access and thread-safe with no
/// shared seek position. Headers are parsed once up front; tensor data is never held resident.
public final class RangedFileWeightSource: WeightSource, @unchecked Sendable {
    private struct Located { let fd: Int32; let entry: SafetensorsEntry }

    private let fds: [Int32]
    private let map: [String: Located]
    public let isStreaming: Bool
    public let freesOnRelease = true

    /// Open one or more `.safetensors` files and index their headers. Duplicate tensor keys across
    /// files throw (matching `SafetensorsWeightSource`), so generic names like `norm.weight` from
    /// different components can't silently cross-wire.
    public init(files: [URL], isStreaming: Bool = false) throws {
        guard !files.isEmpty else { throw WeightSourceError.noFiles }
        var fds: [Int32] = []
        var map: [String: Located] = [:]
        do {
            for url in files {
                let fd = open(url.path, O_RDONLY)
                guard fd >= 0 else { throw WeightSourceError.cannotOpen(url.path) }
                fds.append(fd)
                for (name, entry) in try SafetensorsHeader.parse(url: url) {
                    guard map[name] == nil else { throw WeightSourceError.duplicateTensor(name) }
                    map[name] = Located(fd: fd, entry: entry)
                }
            }
        } catch {
            for fd in fds { close(fd) }   // don't leak fds if a later file fails to parse
            throw error
        }
        self.fds = fds
        self.map = map
        self.isStreaming = isStreaming
    }

    deinit { for fd in fds { close(fd) } }

    public func tensor(_ key: TensorKey) throws -> MLXArray {
        guard let located = map[key.name] else { throw WeightSourceError.missingTensor(key.name) }
        let entry = located.entry
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: max(entry.byteCount, 1), alignment: 64)

        var read = 0
        while read < entry.byteCount {
            let n = pread(located.fd, buffer.advanced(by: read), entry.byteCount - read, off_t(entry.offset + read))
            guard n > 0 else { buffer.deallocate(); throw WeightSourceError.shortRead(key.name) }
            read += n
        }
        // MLX takes ownership of `buffer` and calls the finalizer to free it when the array (and
        // any views) are released — so dropping a streamed block reclaims its weights.
        return MLXArray(rawPointer: buffer, entry.shape, dtype: entry.dtype, finalizer: { buffer.deallocate() })
    }

    public var tensorNames: [String] { Array(map.keys) }
    public var count: Int { map.count }
}
