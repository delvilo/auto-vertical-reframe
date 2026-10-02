# Bolt's Journal

Critical learnings on codebase performance, unexpected bottlenecks, and performance patterns.

## 2025-03-30 - Spectral Residual Complex Spectrum Scaling
**Learning:** Reconstructing the complex DFT spectrum from magnitude and phase using `cv2.phase(real, imag)` (`atan2`) followed by `np.cos(phase)` and `np.sin(phase)` incurs heavy elementwise trigonometric overhead across 2D spatial frequency grids. Since $\cos(\phi) = \text{real} / M$ and $\sin(\phi) = \text{imag} / M$, the reconstruction simplifies to direct magnitude scaling: `scale = np.exp(spectral_residual) / (magnitude + 1e-6)`.
**Action:** When working with DFT magnitudes and phases, replace explicit `atan2`, `cos`, and `sin` calls with direct algebraic ratio scaling.
