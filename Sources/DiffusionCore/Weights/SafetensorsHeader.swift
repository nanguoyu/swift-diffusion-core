import Foundation
import MLX

/// Where a single tensor lives inside a `.safetensors` file, and how to interpret its bytes.
struct SafetensorsEntry {
    let dtype: DType
    let shape: [Int]
    let offset: Int      // absolute byte offset of the tensor's data within the file
    let byteCount: Int
}

/// Minimal safetensors header reader. The format is: 8-byte little-endian `u64` header length,
/// then that many bytes of JSON `{ name: { dtype, shape, data_offsets:[start,end] }, … }`, then a
/// contiguous data region. A tensor's bytes are `[8 + headerLen + start, 8 + headerLen + end)`.
///
/// Reading only the header lets `RangedFileWeightSource` `pread` each tensor's exact range on
/// demand instead of materializing the whole file — the basis of block streaming.
enum SafetensorsHeader {
    static func parse(url: URL) throws -> [String: SafetensorsEntry] {
        let path = url.path
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            throw WeightSourceError.cannotOpen(path)
        }
        defer { try? handle.close() }

        guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
            throw WeightSourceError.malformedHeader(path)
        }
        let headerLength = Int(UInt64(littleEndian: lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }))
        guard headerLength > 0,
              let jsonData = try handle.read(upToCount: headerLength), jsonData.count == headerLength,
              let root = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any]
        else { throw WeightSourceError.malformedHeader(path) }

        let dataBase = 8 + headerLength
        var entries: [String: SafetensorsEntry] = [:]
        entries.reserveCapacity(root.count)
        for (name, value) in root {
            if name == "__metadata__" { continue }
            guard let field = value as? [String: Any],
                  let dtypeName = field["dtype"] as? String,
                  let rawShape = field["shape"] as? [Any],
                  let offsets = field["data_offsets"] as? [Any], offsets.count == 2,
                  let start = (offsets[0] as? NSNumber)?.intValue,
                  let end = (offsets[1] as? NSNumber)?.intValue
            else { throw WeightSourceError.malformedHeader(path) }

            let shape = rawShape.compactMap { ($0 as? NSNumber)?.intValue }
            entries[name] = SafetensorsEntry(dtype: try dtype(from: dtypeName),
                                             shape: shape,
                                             offset: dataBase + start,
                                             byteCount: end - start)
        }
        return entries
    }

    /// Map a safetensors dtype string to an MLX `DType`. The raw little-endian byte layout matches
    /// what MLX expects, so no conversion is needed beyond tagging the type.
    static func dtype(from name: String) throws -> DType {
        switch name {
        case "F64": return .float64
        case "F32": return .float32
        case "F16": return .float16
        case "BF16": return .bfloat16
        case "I64": return .int64
        case "I32": return .int32
        case "I16": return .int16
        case "I8": return .int8
        case "U64": return .uint64
        case "U32": return .uint32
        case "U16": return .uint16
        case "U8": return .uint8
        case "BOOL": return .bool
        default: throw WeightSourceError.unsupportedDType(name)
        }
    }
}
