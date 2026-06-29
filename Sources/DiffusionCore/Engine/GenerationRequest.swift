import CoreGraphics
import Foundation

/// Cooperative control for one in-flight generation.
///
/// Engines call `checkpoint()` at safe boundaries. `pause()` blocks the next checkpoint without
/// trying to serialize MLX state; `cancel()` wakes any paused checkpoint and throws
/// `CancellationError`. This keeps pause/resume session-only and step-boundary scoped.
public final class GenerationControl: @unchecked Sendable {
    private let condition = NSCondition()
    private var paused = false
    private var cancelled = false

    public init() {}

    public var isPaused: Bool {
        condition.lock(); defer { condition.unlock() }
        return paused
    }

    public func pause() {
        condition.lock(); paused = true; condition.unlock()
    }

    public func resume() {
        condition.lock()
        paused = false
        condition.broadcast()
        condition.unlock()
    }

    public func cancel() {
        condition.lock()
        cancelled = true
        paused = false
        condition.broadcast()
        condition.unlock()
    }

    public func checkpoint() throws {
        try Task.checkCancellation()
        condition.lock()
        while paused && !cancelled { condition.wait() }
        let shouldCancel = cancelled
        condition.unlock()
        if shouldCancel { throw CancellationError() }
        try Task.checkCancellation()
    }
}

/// A single text-to-image (or image-to-image) request.
public struct GenerationRequest: Sendable {
    public var prompt: String
    /// Reserved for classifier-free guidance. CFG is not yet implemented; the shipped distilled
    /// models (Z-Image Turbo, FLUX.2 Klein) run guidance-free, so this is currently ignored.
    public var negativePrompt: String?
    public var steps: Int
    /// Reserved — see `negativePrompt`. Distilled models use guidance 1.0; currently ignored.
    public var guidance: Float
    public var seed: UInt64
    public var size: ImageSize
    /// Optional reference image for classic (strength-based) image-to-image. `referenceImages` takes
    /// precedence when non-empty.
    public var referenceImage: CGImage?
    /// Reference images for FLUX.2-style reference-CONTEXT i2i (1–3 images encoded and concatenated
    /// into the transformer sequence as conditioning; output denoised from pure noise). Empty = no i2i.
    public var referenceImages: [CGImage]
    public var strength: Float
    /// Optional cooperative control used for cancel/pause at engine-defined safe points.
    public var control: GenerationControl?

    public init(prompt: String,
                negativePrompt: String? = nil,
                steps: Int,
                guidance: Float = 1.0,
                seed: UInt64,
                size: ImageSize = .square1024,
                referenceImage: CGImage? = nil,
                referenceImages: [CGImage] = [],
                strength: Float = 0.6,
                control: GenerationControl? = nil) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.size = size
        self.referenceImage = referenceImage
        self.referenceImages = referenceImages
        self.strength = strength
        self.control = control
    }
}

public struct ImageSize: Sendable, Hashable {
    public var width: Int
    public var height: Int
    public init(width: Int, height: Int) { self.width = width; self.height = height }

    public static let square512  = ImageSize(width: 512,  height: 512)
    public static let square768  = ImageSize(width: 768,  height: 768)
    public static let square1024 = ImageSize(width: 1024, height: 1024)
}
