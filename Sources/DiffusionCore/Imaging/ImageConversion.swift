@preconcurrency import MLX
import CoreGraphics
import Foundation

/// Value range of a float pixel/latent array.
public enum PixelRange {
    case unit    // [0, 1]
    case signed  // [-1, 1] (typical VAE output)
}

/// Conversions between MLX latents/pixels and `CGImage`.
public enum ImageConversion {

    /// `[H, W, 3]` float MLXArray → 8-bit `CGImage`. Pass `range: .signed` for VAE output in
    /// [-1, 1]; stating the range is required so a [-1, 1] decode can't silently clamp to black.
    public static func cgImage(fromHWC array: MLXArray, range: PixelRange = .unit) -> CGImage? {
        let shape = array.shape
        guard shape.count == 3, shape[2] == 3 else { return nil }
        let h = shape[0], w = shape[1]
        let unit = (range == .signed) ? denormalize(array) : array
        let floats = unit.asArray(Float.self)
        guard floats.count == h * w * 3 else { return nil }
        var rgba = [UInt8](repeating: 255, count: h * w * 4)
        for i in 0..<(h * w) {
            for c in 0..<3 {
                let v = floats[i * 3 + c]
                rgba[i * 4 + c] = UInt8(max(0, min(255, (v * 255).rounded())))
            }
        }
        let space = CGColorSpaceCreateDeviceRGB()
        // Alpha is always opaque (255); declare it skipped so the bytes are straight RGB.
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: w * 4, space: space, bitmapInfo: info, provider: provider,
                       decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }

    /// Map a `[-1, 1]` array (typical VAE output range) to `[0, 1]`.
    public static func denormalize(_ array: MLXArray) -> MLXArray { (array + 1) / 2 }

    /// `CGImage` → `[H, W, 3]` float MLXArray in `[0, 1]` (for image-to-image). Alpha is
    /// intentionally flattened: the source is treated as opaque and read back as straight RGB.
    public static func mlxArray(from image: CGImage) -> MLXArray? {
        let w = image.width, h = image.height
        guard w > 0, h > 0 else { return nil }
        var buf = [UInt8](repeating: 0, count: w * h * 4)
        let space = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: space,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        var floats = [Double](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            for c in 0..<3 { floats[i * 3 + c] = Double(buf[i * 4 + c]) / 255 }
        }
        return MLXArray(converting: floats, [h, w, 3]).asType(.float32)
    }
}
