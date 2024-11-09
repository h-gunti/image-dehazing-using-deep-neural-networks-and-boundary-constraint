from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import os

# Define paths to your dehazed and ground truth image folders
dehazed_folder = 'NH-Haze pred'
ground_truth_folder = 'NH-Haze GT'

psnr_values = []
ssim_values = []

# Iterate over image pairs
for filename in os.listdir(dehazed_folder):
    dehazed_image = cv2.imread(os.path.join(dehazed_folder, filename))
    ground_truth_image = cv2.imread(os.path.join(ground_truth_folder, filename))

    # Ensure images are loaded
    if dehazed_image is not None and ground_truth_image is not None:
        # Convert images to grayscale if needed
        dehazed_image_gray = cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2GRAY)
        ground_truth_image_gray = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate PSNR and SSIM
        psnr_value = psnr(ground_truth_image_gray, dehazed_image_gray)
        ssim_value = ssim(ground_truth_image_gray, dehazed_image_gray)
        
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

# Calculate the average PSNR and SSIM
avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

print(f"Average PSNR: {avg_psnr:.2f}")
print(f"Average SSIM: {avg_ssim:.4f}")
