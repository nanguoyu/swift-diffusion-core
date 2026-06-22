import XCTest
@preconcurrency import MLX
@testable import DiffusionCore

final class SamplerTests: XCTestCase {
    func testFlowMatchTimestepsLinear() {
        let ts = FlowMatchEulerSampler(shift: 1.0).timesteps(steps: 4)
        XCTAssertEqual(ts.count, 5)
        XCTAssertEqual(ts.first!, 1.0, accuracy: 1e-6)
        XCTAssertEqual(ts.last!, 0.0, accuracy: 1e-6)
        XCTAssertEqual(ts[2], 0.5, accuracy: 1e-6)
        for i in 1..<ts.count { XCTAssertLessThan(ts[i], ts[i - 1]) }
    }

    func testShiftSkewsScheduleHigher() {
        let ts = FlowMatchEulerSampler(shift: 3.0).timesteps(steps: 4)
        XCTAssertEqual(ts.first!, 1.0, accuracy: 1e-6)
        XCTAssertEqual(ts.last!, 0.0, accuracy: 1e-6)
        for i in 1..<ts.count { XCTAssertLessThan(ts[i], ts[i - 1]) }
        XCTAssertGreaterThan(ts[2], 0.5)   // shift > 1 keeps mid sigmas noisier than linear
    }

    func testEulerStep() {
        let latent = MLXArray(converting: [0.0], [1]).asType(.float32)
        let velocity = MLXArray(converting: [1.0], [1]).asType(.float32)
        let out = FlowMatchEulerSampler().step(latent: latent, modelOutput: velocity, t: 1.0, tPrev: 0.75)
        XCTAssertEqual(out.item(Float.self), -0.25, accuracy: 1e-5)   // 0 + 1 * (0.75 - 1)
    }
}

final class GovernorTests: XCTestCase {
    private func variant(_ t: Double, _ e: Double, _ v: Double) -> ModelVariant {
        ModelVariant(precision: .q4, approximateBytes: Int64((t + e + v) * 1e9),
                     components: ComponentSizes(transformer: Int64(t * 1e9),
                                                textEncoder: Int64(e * 1e9), vae: Int64(v * 1e9)),
                     layout: .mfluxShard, source: ModelSource(huggingFaceRepo: "example/m"))
    }
    private let mac = DeviceTier(physicalMemoryBytes: 18_000_000_000, isPhone: false)
    private let ip8 = DeviceTier(physicalMemoryBytes: 8_000_000_000, isPhone: true)
    private let ip12 = DeviceTier(physicalMemoryBytes: 12_000_000_000, isPhone: true)

    func testKlein4bRunsResidentEverywhere() {
        let klein4b = variant(2.18, 2.26, 0.17)
        XCTAssertEqual(MemoryGovernor.plan(variant: klein4b, device: mac, externalSSDAvailable: false).residency, .resident)
        XCTAssertEqual(MemoryGovernor.plan(variant: klein4b, device: ip8, externalSSDAvailable: false).residency, .resident)
        XCTAssertEqual(MemoryGovernor.plan(variant: klein4b, device: ip12, externalSSDAvailable: false).residency, .resident)
    }

    func testBigTransformerStreamsOnPhone() {
        let big = variant(5.0, 2.3, 0.16)
        XCTAssertEqual(MemoryGovernor.plan(variant: big, device: ip8, externalSSDAvailable: false).residency, .streamingInternal)
        XCTAssertEqual(MemoryGovernor.plan(variant: big, device: ip8, externalSSDAvailable: true).residency, .streamingExternal)
    }

    func testHugeEncoderUnsupportedOnPhone() {
        let qwen = variant(11.5, 14.14, 0.25)   // 14 GB encoder alone
        XCTAssertFalse(MemoryGovernor.plan(variant: qwen, device: ip8, externalSSDAvailable: true).runnable)
    }
}

final class ImageTests: XCTestCase {
    func testCGImageRoundTrip() throws {
        let arr = MLXArray(converting: [Double](repeating: 0.5, count: 2 * 2 * 3), [2, 2, 3]).asType(.float32)
        let image = try XCTUnwrap(ImageConversion.cgImage(fromHWC: arr))
        XCTAssertEqual(image.width, 2)
        XCTAssertEqual(image.height, 2)
        let back = try XCTUnwrap(ImageConversion.mlxArray(from: image))
        XCTAssertEqual(back.shape, [2, 2, 3])
    }

    func testWrongShapeReturnsNil() {
        let arr = MLXArray(converting: [0.0, 0.0], [2]).asType(.float32)
        XCTAssertNil(ImageConversion.cgImage(fromHWC: arr))
    }
}
