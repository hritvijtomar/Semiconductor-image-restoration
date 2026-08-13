# Phase 4 Summary: RCAN training optimization

## Objective

The objective of Phase 4 was to improve the RCAN super-resolution model through better training methodology and data augmentation. The baseline RCAN model from Phase 3 achieved **25.79 dB PSNR** on the validation set.

## Improvements implemented

### Data augmentation

Training-time augmentation was added to improve generalization.

Augmentations applied:

* Random horizontal flip
* Random vertical flip
* Random 90° rotation
* Random 64×64 crop

Validation images were **not augmented** to ensure unbiased evaluation.

### Extended training

The training duration was increased from **10 epochs to 30 epochs**.

### Learning rate scheduling

The previous **StepLR** scheduler was replaced with **CosineAnnealingLR** with:

* Initial learning rate: **1e-4**
* Minimum learning rate: **1e-6**
* Cosine decay over 30 epochs

### Experiment tracking

The training pipeline was updated to save:

* Best checkpoint (`best_rcan_v2.pth`)
* Last checkpoint (`last_rcan_v2.pth`)
* Training history (`training_history_v2.pth`)
* Total training time

## Results

| Configuration                                         | Validation PSNR |
| ----------------------------------------------------- | --------------- |
| RCAN v1 (10 epochs)                                   | **25.79 dB**    |
| RCAN v2 (30 epochs + augmentation + cosine scheduler) | **25.90 dB**    |

**Improvement:** **+0.11 dB**

Total training time: **32.85 minutes**

## Training analysis

Training and validation curves showed stable convergence.

Observations:

* Training loss decreased steadily.
* Validation loss decreased consistently.
* Validation PSNR increased throughout training.
* No significant overfitting was observed.
* The model converged near epoch 30.

The best validation performance was achieved at **epoch 30**, indicating that the cosine learning rate schedule allowed continued refinement during the later stages of training.

## Deliverables

Phase 4 produced:

* Optimized RCAN model (`best_rcan_v2.pth`)
* Training history
* Training and validation plots
* Experiment documentation

## Conclusion

Phase 4 successfully improved the RCAN model through augmentation and optimized training. The model achieved **25.90 dB PSNR**, representing the best validation performance obtained in the project so far and providing a stronger baseline for the Phase 5 evaluation pipeline.
