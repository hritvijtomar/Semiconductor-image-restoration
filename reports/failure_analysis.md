# Phase 5 failure-case analysis

## RCAN strengths

RCAN performs well on images containing clear structural information such as edges, boundaries, and elongated patterns. The model effectively suppresses noise while preserving the overall geometry of objects. In the leaf and feather examples, the restored images exhibit smoother regions and sharper structural boundaries compared with the noisy inputs.

## RCAN limitations

RCAN struggles with extremely dense textures and very low-contrast regions. In images containing complex fine-scale patterns, the model tends to oversmooth the output, resulting in partial loss of high-frequency details. Similarly, weak contrast regions may remain blurred after restoration because the underlying signal is difficult to distinguish from noise.

## Interpretation

These failure cases are consistent with the behavior of supervised restoration networks trained with pixel-based losses. The model prioritizes noise suppression and overall structural fidelity, which can lead to reduced recovery of very fine textures when the degradation is severe.

## Conclusion

RCAN provides a favorable trade-off between denoising and structural preservation, but further improvements could be achieved through perceptual losses, larger RCAN variants, or additional training on more diverse degradation patterns.
