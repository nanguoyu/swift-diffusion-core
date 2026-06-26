import XCTest
@preconcurrency import MLX
@testable import DiffusionCore

final class SamplerTests: XCTestCase {
    func testFlowMatchTimestepsLinear() {
        let ts = FlowMatchEulerSampler(shift: 1.0).timesteps(steps: 4)
        XCTAssertEqual(ts.count, 5)
        XCTAssertEqual(ts.first!, 1.0, accuracy: 1e-6)
        XCTAssertEqual(ts.last!, 0.0, accuracy: 1e-6)
        // Working sigmas span 1 → sigma_min (1/1000), not 1 → 0; a separate 0 is appended.
        XCTAssertEqual(ts[2], 0.334, accuracy: 1e-2)
        XCTAssertEqual(ts[3], 0.001, accuracy: 1e-3)   // penultimate = sigma_min, not 0.25
        for i in 1..<ts.count { XCTAssertLessThan(ts[i], ts[i - 1]) }
    }

    func testShiftSkewsScheduleHigher() {
        let ts = FlowMatchEulerSampler(shift: 3.0).timesteps(steps: 4)
        XCTAssertEqual(ts.first!, 1.0, accuracy: 1e-6)
        XCTAssertEqual(ts.last!, 0.0, accuracy: 1e-6)
        for i in 1..<ts.count { XCTAssertLessThan(ts[i], ts[i - 1]) }
        XCTAssertGreaterThan(ts[2], 0.5)   // shift > 1 keeps mid sigmas noisier than linear
        // The final Euler step lands on a tiny shifted sigma_min, not a 0.5 cliff (the old bug).
        XCTAssertLessThan(ts[ts.count - 2], 0.05)
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

    // MARK: - Sequence-aware working set (Wave-1 step 4)

    func testWorkingSetScalesLinearlyWithSeqLen() {
        // Reference (512 px, 1024 tokens) = the 1 GB base; 1024 px (4096 tokens) = 4×; below the
        // reference is clamped to the base so ≤512 plans never shrink.
        XCTAssertEqual(MemoryGovernor.workingSet(forImageSeqLen: 1024), 1_000_000_000)
        XCTAssertEqual(MemoryGovernor.workingSet(forImageSeqLen: 4096), 4_000_000_000)
        XCTAssertEqual(MemoryGovernor.workingSet(forImageSeqLen: 256), 1_000_000_000)   // clamped up
    }

    func testNilSeqLenMatchesReference() {
        // The default (nil) must be byte-for-byte the old flat-1 GB behavior so no existing caller's
        // plan changes — guards against the audit's "over-conservatism forces streaming at 512" risk.
        let klein4b = variant(2.18, 2.26, 0.17)
        let byDefault = MemoryGovernor.plan(variant: klein4b, device: ip8, externalSSDAvailable: false)
        let byReference = MemoryGovernor.plan(variant: klein4b, device: ip8, externalSSDAvailable: false,
                                              imageSeqLen: MemoryGovernor.referenceImageSeqLen)
        XCTAssertEqual(byDefault.residency, byReference.residency)
        XCTAssertEqual(byDefault.estimatedPeakBytes, byReference.estimatedPeakBytes)
        XCTAssertEqual(byDefault.residency, .resident)   // 512 stays resident on an 8 GB phone
    }

    func testLargeImageNotResidentOnPhone() {
        // The actual 1024 OOM driver: a 4× activation working set no longer fits resident on an 8 GB
        // phone. Must demote off `.resident` (streaming or refuse) BEFORE jetsam.
        let klein4b = variant(2.18, 2.26, 0.17)
        let plan = MemoryGovernor.plan(variant: klein4b, device: ip8, externalSSDAvailable: false,
                                       imageSeqLen: 4096)
        XCTAssertNotEqual(plan.residency, .resident)
    }

    func testLargeImageStillResidentOnMac() {
        // A plugged-in Mac has the headroom — 1024 must stay resident (byte-for-byte fast path).
        let klein4b = variant(2.18, 2.26, 0.17)
        let plan = MemoryGovernor.plan(variant: klein4b, device: mac, externalSSDAvailable: false,
                                       imageSeqLen: 4096)
        XCTAssertEqual(plan.residency, .resident)
    }
}

final class SigmaHookTests: XCTestCase {
    /// The default sigma hook must be byte-for-byte the engine's sampler, so an architecture that
    /// doesn't override it (Z-Image) keeps the exact schedule it had before the hook existed.
    func testDefaultSigmasDelegateToSampler() {
        let arch = MockArchitecture(blocks: 1)
        let sampler = FlowMatchEulerSampler(shift: 3.0, shiftTerminal: 0.02)
        for steps in [2, 4, 8] {
            XCTAssertEqual(arch.sigmas(size: .square1024, steps: steps, sampler: sampler),
                           sampler.timesteps(steps: steps),
                           "default sigmas must equal sampler.timesteps for \(steps) steps")
        }
    }

    /// The default ignores image size (the current samplers are size-independent), so a size change
    /// must not perturb a non-overriding architecture's schedule.
    func testDefaultSigmasAreSizeIndependent() {
        let arch = MockArchitecture(blocks: 1)
        let sampler = FlowMatchEulerSampler(shift: 1.0)
        XCTAssertEqual(arch.sigmas(size: .square512, steps: 4, sampler: sampler),
                       arch.sigmas(size: .square1024, steps: 4, sampler: sampler))
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

    func testSignedRangeDecodes() {
        // [-1, 1] VAE range must map through denormalize, not clamp negatives to black.
        let arr = MLXArray(converting: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], [1, 2, 3]).asType(.float32)
        let image = ImageConversion.cgImage(fromHWC: arr, range: .signed)
        XCTAssertEqual(image?.width, 2)
        XCTAssertEqual(image?.height, 1)
    }
}

final class CapabilitiesTests: XCTestCase {
    private let variant = ModelCatalog.fluxKlein4B.variants.first { $0.precision == .q4 }!

    func testFluxRunnableOnPhone() {
        // FLUX.2 now runs on iPhone: 512 resident via the facade, 1024 block-streamed via the generic
        // engine. capabilities() is memory-driven and no longer hard-blocks the phone.
        let phone = DeviceTier(physicalMemoryBytes: 12_000_000_000, isPhone: true)
        let caps = MLXDiffusionEngine.capabilities(for: ModelCatalog.fluxKlein4B, variant: variant, on: phone)
        XCTAssertTrue(caps.runnable)
        XCTAssertNotEqual(caps.residency, .unsupported)
    }

    func testFluxRunnableOnMac() {
        let mac = DeviceTier(physicalMemoryBytes: 18_000_000_000, isPhone: false)
        let caps = MLXDiffusionEngine.capabilities(for: ModelCatalog.fluxKlein4B, variant: variant, on: mac)
        XCTAssertTrue(caps.runnable)
    }
}
