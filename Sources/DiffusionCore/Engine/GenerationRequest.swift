import CoreGraphics

/// A single text-to-image (or image-to-image) request.
public struct GenerationRequest: Sendable {
    public var prompt: String
    public var negativePrompt: String?
    public var steps: Int
    public var guidance: Float
    public var seed: UInt64
    public var size: ImageSize
    /// Optional reference image for image-to-image.
    public var referenceImage: CGImage?
    public var strength: Float

    public init(prompt: String,
                negativePrompt: String? = nil,
                steps: Int,
                guidance: Float = 1.0,
                seed: UInt64,
                size: ImageSize = .square1024,
                referenceImage: CGImage? = nil,
                strength: Float = 0.6) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.size = size
        self.referenceImage = referenceImage
        self.strength = strength
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
