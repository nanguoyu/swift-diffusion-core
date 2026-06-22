// swift-tools-version: 5.10
import PackageDescription

// swift-diffusion-core — the platform-agnostic heart of the universal diffusion app.
// Engine protocol, streaming/partial-load runtime, WeightSource, samplers, memory governor,
// and the model catalog. Pure MLX + Swift; runs on macOS 14+ and iOS 17+.
//
// Intended to live as its own public repo (nanguoyu/swift-diffusion-core). During local
// development it is consumed as a path dependency from the app under `Packages/`.
let package = Package(
    name: "swift-diffusion-core",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "DiffusionCore", targets: ["DiffusionCore"]),
    ],
    dependencies: [
        // Apple MLX for Swift — the single inference backend (Mac + iPhone).
        // NOTE: pin the exact version during Phase 0.
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.0"),
    ],
    targets: [
        .target(
            name: "DiffusionCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
    ]
)
