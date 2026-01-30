#!/usr/bin/env python3
"""
Lab Exercise 13 - Complete Solutions
Geometric and Intensity Transformations
"""

import scipy as sc
from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import skimage
from skimage import transform, data, img_as_float, exposure

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Create output directory for results
import os
os.makedirs('lab13_results', exist_ok=True)

print("="*70)
print("LAB EXERCISE 13 - SOLUTIONS")
print("="*70)

# Load the ascent image
try:
    f = misc.ascent()
except:
    try:
        from scipy.datasets import ascent
        f = ascent()
    except:
        print("Using alternative image loading method...")
        f = data.camera()

print(f"\nOriginal image shape: {f.shape}")
print(f"Original image dtype: {f.dtype}")

# =============================================================================
# QUESTION 1: GEOMETRIC TRANSFORMATIONS
# =============================================================================

print("\n" + "="*70)
print("QUESTION 1: GEOMETRIC TRANSFORMATIONS")
print("="*70)

# -----------------------------------------------------------------------------
# Question 1.1: Horizontal Shear (0.2) + Rescaling (50%)
# -----------------------------------------------------------------------------

print("\n" + "-"*70)
print("Question 1.1: Horizontal Shear by 0.2 + Rescaling by 50%")
print("-"*70)

# Normalize image to [0, 1] range
f_norm = f / 255.0

# Step 1: Apply horizontal shear by 0.2
# Shear matrix: [[1, shear, 0], [0, 1, 0], [0, 0, 1]]
shear_value = 0.2
shear_matrix = np.array([
    [1, shear_value, 0],
    [0, 1, 0],
    [0, 0, 1]
])

print(f"\nStep 1: Horizontal Shear")
print(f"  Shear value: {shear_value}")
print(f"  Shear matrix:\n{shear_matrix}")

# Create AffineTransform and apply shear
tf_shear = transform.AffineTransform(matrix=shear_matrix)
sheared_image = transform.warp(f_norm, tf_shear, order=1, preserve_range=True, mode='constant')

print(f"  Sheared image shape: {sheared_image.shape}")

# Step 2: Rescale by 50%
scale_factor = 0.5
rescaled_image = transform.rescale(sheared_image, scale_factor, anti_aliasing=True)

print(f"\nStep 2: Rescaling")
print(f"  Scale factor: {scale_factor} (50%)")
print(f"  Final image shape: {rescaled_image.shape}")

# Display results for Question 1.1
fig1_1, axes1_1 = plt.subplots(1, 4, figsize=(16, 4))

axes1_1[0].imshow(f, cmap='gray')
axes1_1[0].set_title('Original Image\n(ascent)', fontsize=11, fontweight='bold')
axes1_1[0].axis('off')

axes1_1[1].imshow(sheared_image, cmap='gray')
axes1_1[1].set_title(f'After Horizontal Shear\n(shear = {shear_value})', fontsize=11, fontweight='bold')
axes1_1[1].axis('off')

axes1_1[2].imshow(rescaled_image, cmap='gray')
axes1_1[2].set_title(f'After Rescaling\n(scale = {scale_factor})', fontsize=11, fontweight='bold')
axes1_1[2].axis('off')

axes1_1[3].imshow(rescaled_image, cmap='gray')
axes1_1[3].set_title('Final Result\nSheared + Rescaled', fontsize=11, fontweight='bold')
axes1_1[3].axis('off')

plt.suptitle('Question 1.1: Horizontal Shear (0.2) + Rescaling (50%)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('lab13_results/question1_1_shear_rescale.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: lab13_results/question1_1_shear_rescale.png")
plt.close()

# -----------------------------------------------------------------------------
# Question 1.2: Translation (x: +5px, y: +10px)
# -----------------------------------------------------------------------------

print("\n" + "-"*70)
print("Question 1.2: Translation by 5 pixels (x) and 10 pixels (y)")
print("-"*70)

# Define translation parameters
tx = 5   # x-direction (horizontal shift)
ty = 10  # y-direction (vertical shift)

# Create translation matrix
# Format: [[1, 0, tx], [0, 1, ty], [0, 0, 1]]
translation_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty],
    [0, 0, 1]
])

print(f"\nTranslation parameters:")
print(f"  X-direction shift: {tx} pixels")
print(f"  Y-direction shift: {ty} pixels")
print(f"\nTranslation matrix:")
print(translation_matrix)

# Create AffineTransform and apply translation
tf_transl = transform.AffineTransform(matrix=translation_matrix)
translated_image = transform.warp(f_norm, tf_transl, order=1, preserve_range=True, mode='constant')

print(f"\nTranslated image shape: {translated_image.shape}")

# Display results for Question 1.2
fig1_2, axes1_2 = plt.subplots(1, 3, figsize=(15, 5))

axes1_2[0].imshow(f, cmap='gray')
axes1_2[0].set_title('Original Image', fontsize=12, fontweight='bold')
axes1_2[0].axis('off')

axes1_2[1].imshow(translated_image, cmap='gray')
axes1_2[1].set_title(f'Translated Image\n(x: +{tx}px, y: +{ty}px)', fontsize=12, fontweight='bold')
axes1_2[1].axis('off')

# Overlay comparison
axes1_2[2].imshow(f_norm, cmap='Reds', alpha=0.5)
axes1_2[2].imshow(translated_image, cmap='Blues', alpha=0.5)
axes1_2[2].set_title('Overlay Comparison\n(Red=Original, Blue=Translated)', fontsize=12, fontweight='bold')
axes1_2[2].axis('off')

plt.suptitle('Question 1.2: Translation using AffineTransform', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('lab13_results/question1_2_translation.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: lab13_results/question1_2_translation.png")
plt.close()

# -----------------------------------------------------------------------------
# Question 1.3: Rotate (-45°) → Rescale (1.2) → Crop (50% central)
# -----------------------------------------------------------------------------

print("\n" + "-"*70)
print("Question 1.3: Rotate (-45°) → Rescale (1.2) → Crop (50%)")
print("-"*70)

# Step 1: Rotate by -45 degrees
rotation_angle = -45
rotated_image = transform.rotate(f_norm, rotation_angle, resize=True, mode='constant')

print(f"\nStep 1: Rotation")
print(f"  Rotation angle: {rotation_angle}°")
print(f"  Original shape: {f_norm.shape}")
print(f"  Rotated shape: {rotated_image.shape}")

# Step 2: Rescale by 1.2 (120%)
rescale_factor = 1.2
rescaled_rotated = transform.rescale(rotated_image, rescale_factor, anti_aliasing=True)

print(f"\nStep 2: Rescaling")
print(f"  Scale factor: {rescale_factor} (120%)")
print(f"  Rescaled shape: {rescaled_rotated.shape}")

# Step 3: Crop to 50% of central part
height, width = rescaled_rotated.shape
crop_height = int(height * 0.5)
crop_width = int(width * 0.5)

# Calculate crop boundaries (centered)
start_h = (height - crop_height) // 2
end_h = start_h + crop_height
start_w = (width - crop_width) // 2
end_w = start_w + crop_width

# Perform crop
cropped_image = rescaled_rotated[start_h:end_h, start_w:end_w]

print(f"\nStep 3: Cropping")
print(f"  Crop percentage: 50% of central part")
print(f"  Crop dimensions: {crop_height} x {crop_width}")
print(f"  Final shape: {cropped_image.shape}")

# Display results for Question 1.3
fig1_3, axes1_3 = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Transformation steps
axes1_3[0, 0].imshow(f_norm, cmap='gray')
axes1_3[0, 0].set_title('Step 0: Original Image', fontsize=11, fontweight='bold')
axes1_3[0, 0].axis('off')

axes1_3[0, 1].imshow(rotated_image, cmap='gray')
axes1_3[0, 1].set_title(f'Step 1: Rotated ({rotation_angle}°)\nShape: {rotated_image.shape}', 
                         fontsize=11, fontweight='bold')
axes1_3[0, 1].axis('off')

axes1_3[0, 2].imshow(rescaled_rotated, cmap='gray')
axes1_3[0, 2].set_title(f'Step 2: Rescaled ({rescale_factor}x)\nShape: {rescaled_rotated.shape}', 
                         fontsize=11, fontweight='bold')
axes1_3[0, 2].axis('off')

# Row 2: Crop visualization and final result
axes1_3[1, 0].imshow(rescaled_rotated, cmap='gray')
from matplotlib.patches import Rectangle
rect = Rectangle((start_w, start_h), crop_width, crop_height, 
                  linewidth=3, edgecolor='red', facecolor='none')
axes1_3[1, 0].add_patch(rect)
axes1_3[1, 0].set_title('Step 3: Crop Area\n(Red Rectangle = 50% Central)', 
                        fontsize=11, fontweight='bold')
axes1_3[1, 0].axis('off')

axes1_3[1, 1].imshow(cropped_image, cmap='gray')
axes1_3[1, 1].set_title(f'Final Result\nShape: {cropped_image.shape}', 
                        fontsize=11, fontweight='bold')
axes1_3[1, 1].axis('off')

# Comparison
axes1_3[1, 2].imshow(cropped_image, cmap='gray')
axes1_3[1, 2].set_title('Final: Rotated +\nRescaled + Cropped', 
                        fontsize=11, fontweight='bold')
axes1_3[1, 2].axis('off')

plt.suptitle('Question 1.3: Sequential Operations (Rotate → Rescale → Crop)', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('lab13_results/question1_3_rotate_rescale_crop.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: lab13_results/question1_3_rotate_rescale_crop.png")
plt.close()

# =============================================================================
# QUESTION 2: GAMMA CORRECTION AND LOGARITHMIC TRANSFORMATION
# =============================================================================

print("\n" + "="*70)
print("QUESTION 2: INTENSITY TRANSFORMATIONS")
print("="*70)

# Load moon image
im = data.moon()
im = img_as_float(im)

print(f"\nMoon image shape: {im.shape}")
print(f"Moon image range: [{im.min():.3f}, {im.max():.3f}]")

# -----------------------------------------------------------------------------
# Question 2.1: Gamma Correction Effects
# -----------------------------------------------------------------------------

print("\n" + "-"*70)
print("Question 2.1: Gamma Correction Effects")
print("-"*70)

# Original image
# Gamma corrected (γ = 2, makes image darker)
gamma_2 = exposure.adjust_gamma(im, 2)
# Gamma corrected (γ < 1, makes image brighter)
gamma_0_5 = exposure.adjust_gamma(im, 0.5)

print("\nGamma transformations performed:")
print(f"  γ = 2.0  : Darkens the image (compresses high intensities)")
print(f"  γ = 0.5  : Brightens the image (expands low intensities)")

# -----------------------------------------------------------------------------
# Question 2.2: Logarithmic Transformation
# -----------------------------------------------------------------------------

print("\n" + "-"*70)
print("Question 2.2: Logarithmic Transformation")
print("-"*70)

# Logarithmic transformation
log_corrected = exposure.adjust_log(im, 1)

print("\nLogarithmic transformation effects:")
print("  - Expands low intensity values (brightens dark regions)")
print("  - Compresses high intensity values")
print("  - Enhances details in dark areas")
print("  - Useful for images with large dynamic range")

# -----------------------------------------------------------------------------
# Question 2.3: Compare Different Gamma Values
# -----------------------------------------------------------------------------

print("\n" + "-"*70)
print("Question 2.3: Comparison of Gamma Values")
print("-"*70)

print("\nComparison of gamma transformations:")
print(f"  γ < 1 (0.5): Brightens image, expands shadows, compresses highlights")
print(f"  γ = 1:       No change (identity transformation)")
print(f"  γ > 1 (2.0): Darkens image, compresses shadows, expands highlights")

# Create comprehensive visualization
fig2, axes = plt.subplots(3, 4, figsize=(16, 12))

# Row 1: Original and transformations
axes[0, 0].imshow(im, cmap='gray')
axes[0, 0].set_title('Original Image', fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(gamma_2, cmap='gray')
axes[0, 1].set_title('Gamma = 2.0\n(Darker)', fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(gamma_0_5, cmap='gray')
axes[0, 2].set_title('Gamma = 0.5\n(Brighter)', fontsize=11, fontweight='bold')
axes[0, 2].axis('off')

axes[0, 3].imshow(log_corrected, cmap='gray')
axes[0, 3].set_title('Logarithmic\nTransformation', fontsize=11, fontweight='bold')
axes[0, 3].axis('off')

# Row 2: Histograms
axes[1, 0].hist(im.ravel(), bins=50, histtype='stepfilled', color='blue', alpha=0.7)
axes[1, 0].set_title('Histogram: Original', fontsize=10)
axes[1, 0].set_xlabel('Pixel Intensity')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].hist(gamma_2.ravel(), bins=50, histtype='stepfilled', color='red', alpha=0.7)
axes[1, 1].set_title('Histogram: γ = 2.0', fontsize=10)
axes[1, 1].set_xlabel('Pixel Intensity')
axes[1, 1].grid(alpha=0.3)

axes[1, 2].hist(gamma_0_5.ravel(), bins=50, histtype='stepfilled', color='green', alpha=0.7)
axes[1, 2].set_title('Histogram: γ = 0.5', fontsize=10)
axes[1, 2].set_xlabel('Pixel Intensity')
axes[1, 2].grid(alpha=0.3)

axes[1, 3].hist(log_corrected.ravel(), bins=50, histtype='stepfilled', color='orange', alpha=0.7)
axes[1, 3].set_title('Histogram: Log Transform', fontsize=10)
axes[1, 3].set_xlabel('Pixel Intensity')
axes[1, 3].grid(alpha=0.3)

# Row 3: Transformation curves
x = np.linspace(0, 1, 256)

axes[2, 0].plot(x, x, 'b-', linewidth=2, label='Identity (γ=1)')
axes[2, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[2, 0].set_title('Identity Transform', fontsize=10)
axes[2, 0].set_xlabel('Input Intensity')
axes[2, 0].set_ylabel('Output Intensity')
axes[2, 0].grid(alpha=0.3)
axes[2, 0].legend()

axes[2, 1].plot(x, x**2, 'r-', linewidth=2, label='γ = 2.0')
axes[2, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[2, 1].set_title('Gamma = 2.0 Curve', fontsize=10)
axes[2, 1].set_xlabel('Input Intensity')
axes[2, 1].set_ylabel('Output Intensity')
axes[2, 1].grid(alpha=0.3)
axes[2, 1].legend()

axes[2, 2].plot(x, x**0.5, 'g-', linewidth=2, label='γ = 0.5')
axes[2, 2].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[2, 2].set_title('Gamma = 0.5 Curve', fontsize=10)
axes[2, 2].set_xlabel('Input Intensity')
axes[2, 2].set_ylabel('Output Intensity')
axes[2, 2].grid(alpha=0.3)
axes[2, 2].legend()

axes[2, 3].plot(x, np.log1p(x) / np.log1p(1), 'orange', linewidth=2, label='Log')
axes[2, 3].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[2, 3].set_title('Logarithmic Curve', fontsize=10)
axes[2, 3].set_xlabel('Input Intensity')
axes[2, 3].set_ylabel('Output Intensity')
axes[2, 3].grid(alpha=0.3)
axes[2, 3].legend()

plt.suptitle('Question 2: Intensity Transformations Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('lab13_results/question2_intensity_transformations.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: lab13_results/question2_intensity_transformations.png")
plt.close()

# =============================================================================
# SUMMARY REPORT
# =============================================================================

print("\n" + "="*70)
print("SUMMARY OF ANSWERS")
print("="*70)

print("\nQUESTION 1: GEOMETRIC TRANSFORMATIONS")
print("-" * 70)
print("\n1.1 Horizontal Shear (0.2) + Rescaling (50%):")
print(f"    - Original shape: {f.shape}")
print(f"    - After shear: {sheared_image.shape}")
print(f"    - Final shape: {rescaled_image.shape}")
print(f"    - Operations: Shear → Rescale")

print("\n1.2 Translation (x: +5px, y: +10px):")
print(f"    - Translation matrix defined: [[1, 0, 5], [0, 1, 10], [0, 0, 1]]")
print(f"    - Final shape: {translated_image.shape}")
print(f"    - Image shifted 5 pixels right, 10 pixels down")

print("\n1.3 Rotate → Rescale → Crop:")
print(f"    - Original shape: {f_norm.shape}")
print(f"    - After rotation (-45°): {rotated_image.shape}")
print(f"    - After rescaling (1.2x): {rescaled_rotated.shape}")
print(f"    - Final (50% crop): {cropped_image.shape}")

print("\nQUESTION 2: INTENSITY TRANSFORMATIONS")
print("-" * 70)

print("\n2.1 Gamma Correction Effects:")
print("    - Gamma > 1 (γ=2): Darkens image, shifts histogram left")
print("    - Gamma < 1 (γ=0.5): Brightens image, shifts histogram right")
print("    - Gamma = 1: No change (identity transformation)")

print("\n2.2 Logarithmic Transformation:")
print("    - Expands dark pixel values (brightens shadows)")
print("    - Compresses bright pixel values")
print("    - Improves visibility of details in dark regions")
print("    - Useful for images with large dynamic range")

print("\n2.3 Gamma Comparison (γ < 1 vs γ = 2):")
print("    - γ = 0.5: Brightens, better for underexposed images")
print("    - γ = 2.0: Darkens, better for overexposed images")
print("    - Opposite effects on histogram distribution")
print("    - Both are power-law transformations: output = input^γ")

print("\n" + "="*70)
print("ALL RESULTS SAVED TO: lab13_results/")
print("="*70)

print("\nOutput files:")
print("  1. question1_1_shear_rescale.png")
print("  2. question1_2_translation.png")
print("  3. question1_3_rotate_rescale_crop.png")
print("  4. question2_intensity_transformations.png")

print("\n" + "="*70)
print("LAB EXERCISE 13 - COMPLETED SUCCESSFULLY!")
print("="*70)
