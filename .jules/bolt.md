# Bolt's Journal

Critical learnings on codebase performance, unexpected bottlenecks, and performance patterns.

## 2025-02-18 - Algebraic simplification of complex DFT spectral residual phase reconstruction
**Learning:** In spectral residual saliency computation, reconstructing complex phase spectrum via `cv2.phase` (elementwise `atan2`) followed by `np.cos` and `np.sin` is redundant when magnitude is already calculated ($\cos(\phi) = \text{real} / \text{mag}$, $\sin(\phi) = \text{imag} / \text{mag}$). Direct algebraic scaling using `factor = exp(spectral_residual) / (magnitude + 1e-6)` avoids 3 transcendental function calls per pixel array.
**Action:** When working with DFT/frequency-domain phase manipulation on 2D matrices, check if trigonometric calls can be simplified to algebraic ratios using magnitude/components.
