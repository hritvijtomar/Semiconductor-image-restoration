# Phase 2 summary

## Dataset
- 3200 paired grayscale images
- LR: 128×128
- HR: 256×256

## Models

### EDSR + L1
- 16 residual blocks
- 64 channels
- Best PSNR: 25.41 dB

### EDSR + Charbonnier
- Best PSNR: 25.68 dB

### RCAN + Charbonnier
- 5 residual groups
- 5 RCABs per group
- 64 channels
- Best PSNR: 25.79 dB

## Best model
RCAN + Charbonnier

## Conclusion
Channel attention produced the best quantitative result, improving the EDSR baseline by 0.38 dB.