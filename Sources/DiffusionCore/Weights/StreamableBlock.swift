import MLX

/// One independently loadable unit of a denoiser — typically one transformer block.
///
/// The core streaming engine drives a sequence of these: load → run → release, so peak
/// weight residency stays bounded. This is the MLX analogue of the split-stage CoreML
/// loading used to fit large transformers on a phone.
public protocol StreamableBlock: AnyObject {
    /// Stable index within the denoiser stack (for prefetch ordering / logging).
    var index: Int { get }

    /// Approximate weight size of this block, bytes (for the governor's resident-set budget).
    var approximateBytes: Int64 { get }

    /// Load this block's weights from `source`.
    func load(from source: WeightSource) throws

    /// Run the block on the current hidden state.
    func callAsFunction(_ x: MLXArray, conditioning: Conditioning, timestep: MLXArray) -> MLXArray

    /// Release this block's weight arrays so MLX can free them.
    func release()
}
