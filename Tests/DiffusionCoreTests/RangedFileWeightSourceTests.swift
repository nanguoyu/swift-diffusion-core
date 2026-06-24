import XCTest
@preconcurrency import MLX
@testable import DiffusionCore

/// Step 1 of the iPhone block-streaming milestone: prove a tensor read by byte range from a
/// `.safetensors` file reconstructs identically to MLX's own resident load — the load-bearing
/// primitive for `RangedFileWeightSource`.
final class RangedFileWeightSourceTests: XCTestCase {

    func testRangedReadMatchesResidentLoad() throws {
        let arrays: [String: MLXArray] = [
            "alpha": MLXArray([Float(1.0), -2.5, 3.25, 0.0, 7.5]),
            "block.0.weight": MLXArray((0 ..< 12).map { Float($0) }).reshaped([3, 4]),
            "gamma": MLXArray([Float(0.5), 1.5, -3.0, 4.0]).reshaped([2, 2]).asType(.bfloat16),
        ]

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ranged-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: arrays, url: url)

        let source = try RangedFileWeightSource(files: [url])
        XCTAssertTrue(source.freesOnRelease, "a ranged source must free on release to enable streaming")
        XCTAssertEqual(Set(source.tensorNames), Set(arrays.keys))

        for (name, expected) in arrays {
            let got = try source.tensor(TensorKey(name))
            XCTAssertEqual(got.shape, expected.shape, "shape for \(name)")
            XCTAssertEqual(got.dtype, expected.dtype, "dtype for \(name)")
            let matches = allClose(got.asType(.float32), expected.asType(.float32), atol: 1e-2).item(Bool.self)
            XCTAssertTrue(matches, "values for \(name)")
        }
    }

    func testMissingTensorThrows() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ranged-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: ["only": MLXArray([Float(1), 2, 3])], url: url)

        let source = try RangedFileWeightSource(files: [url])
        XCTAssertThrowsError(try source.tensor(TensorKey("missing")))
    }
}
