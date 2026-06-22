# swift-diffusion-core

The platform-agnostic core of a universal (macOS + iOS) on-device diffusion app, built on
Apple MLX + Swift. It defines the boundary the app talks to and the streaming/partial-load
runtime that makes large models fit constrained devices.

## What's here

- `Engine/` — `DiffusionEngine` (the app's only boundary), `GenerationRequest`,
  `GenerationProgress`, `EngineCapabilities`.
- `Architecture/` — `DiffusionArchitecture`, the seam each model package implements
  (e.g. `z-image-swift-mlx`, `flux-2-swift-mlx`).
- `Weights/` — `WeightSource` (mmap internal / ranged-read external SSD / hybrid) and
  `StreamableBlock` (load → run → release), the partial-load primitives.
- `Catalog/` — `DiffusionModel` and the model catalog.
- `Memory/` — `DeviceTier`, `MemoryGovernor`, jetsam-accurate `MemoryProbe`.
- `Sampling/` — the `Sampler` protocol.

## Status

Scaffold / protocol surface. Implementations land in Phase 0. See the app's
`docs/BLUEPRINT.md` for the full plan.

## License

Apache-2.0 (intended).
