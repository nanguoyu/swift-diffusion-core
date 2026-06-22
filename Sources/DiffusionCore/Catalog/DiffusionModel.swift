import Foundation

/// A model in the catalog. `id`/`displayName` are user-facing; on-disk repo paths live in
/// `ModelSource` and are never shown in the UI.
public struct DiffusionModel: Identifiable, Sendable {
    public let id: String
    public let displayName: String
    public let family: ModelFamily
    public let publisher: String
    public let summary: String
    public let license: ModelLicense
    public let architecture: ArchitectureSpec
    public let variants: [ModelVariant]

    public init(id: String, displayName: String, family: ModelFamily, publisher: String,
                summary: String, license: ModelLicense, architecture: ArchitectureSpec,
                variants: [ModelVariant]) {
        self.id = id
        self.displayName = displayName
        self.family = family
        self.publisher = publisher
        self.summary = summary
        self.license = license
        self.architecture = architecture
        self.variants = variants
    }
}

/// One precision variant of a model.
public struct ModelVariant: Identifiable, Sendable {
    public var id: String { precision.rawValue }
    public let precision: Precision
    public let approximateBytes: Int64
    public let components: ComponentSizes
    public let layout: ModelLayout
    public let source: ModelSource

    public init(precision: Precision, approximateBytes: Int64, components: ComponentSizes,
                layout: ModelLayout, source: ModelSource) {
        self.precision = precision
        self.approximateBytes = approximateBytes
        self.components = components
        self.layout = layout
        self.source = source
    }
}

/// Per-component on-disk sizes (bytes). Drives the detail view's component breakdown and the
/// memory governor's two-phase peak estimate.
public struct ComponentSizes: Sendable {
    public let transformer: Int64
    public let textEncoder: Int64
    public let vae: Int64
    public init(transformer: Int64, textEncoder: Int64, vae: Int64) {
        self.transformer = transformer
        self.textEncoder = textEncoder
        self.vae = vae
    }
}

public enum ModelFamily: String, Sendable {
    case zImage, flux2, qwenImage
}

/// Precision tiers. `bits` and `label` drive the UI; quantization is consumed natively by
/// MLX (no load-time dequant-to-fp16 expansion).
public enum Precision: String, Sendable, CaseIterable {
    case bf16, q8, q6, q5, q4, q3, q2
    public var label: String {
        switch self {
        case .bf16: return "bf16"
        case .q8: return "8-bit"
        case .q6: return "6-bit"
        case .q5: return "5-bit"
        case .q4: return "4-bit"
        case .q3: return "3-bit"
        case .q2: return "2-bit"
        }
    }
}

/// On-disk weight layout — the loader dispatches on this because layouts differ across
/// families (mflux numbered shards, quanto int blocks, flat single-file, diffusers folders).
public enum ModelLayout: Sendable {
    case mfluxShard
    case quantoInt
    case flatSingle
    case diffusersFolder
}

/// Where to fetch weights. Public sources only; a user-configurable mirror is applied at
/// download time — never a hardcoded private endpoint.
public struct ModelSource: Sendable {
    public let huggingFaceRepo: String
    public let revision: String
    public init(huggingFaceRepo: String, revision: String = "main") {
        self.huggingFaceRepo = huggingFaceRepo
        self.revision = revision
    }
}

public enum ModelLicense: Sendable {
    case apache2
    case mit
    case other(name: String, commercialUse: Bool)

    public var label: String {
        switch self {
        case .apache2: return "Apache-2.0"
        case .mit: return "MIT"
        case .other(let name, _): return name
        }
    }
    public var allowsCommercialUse: Bool {
        switch self {
        case .apache2, .mit: return true
        case .other(_, let ok): return ok
        }
    }
}
