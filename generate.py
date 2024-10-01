from multiprocessing import Pool, cpu_count
import cv2
import os
import numpy as np
from augraphy import *



# Directory paths
input_dir = 'data/clean_base'        # Replace with your actual path
output_clean_dir = 'data/train_cleaned' # Replace with your actual path
output_dirty_dir = 'data/train' # Replace with your actual path

# Create output directories if they don't exist
os.makedirs(output_clean_dir, exist_ok=True)
os.makedirs(output_dirty_dir, exist_ok=True)

# Load clean images
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Define your augmentation sequences as you have in your original code
ink_phase = AugmentationSequence([
    BleedThrough(),
    LowInkRandomLines(),
    InkBleed(),
    ShadowCast(),
    Brightness(brightness_range=(0.7, 0.99)), 
    LightingGradient(),
    #Hollow(),
    #Letterpress()
], p=1)

paper_phase = AugmentationSequence([
    PaperFactory(),
    NoiseTexturize(),
    Folding(),
    SubtleNoise(subtle_range=5), 
    #BadPhotoCopy(blur_noise=1)
], p=1)

post_phase = AugmentationSequence([
    Jpeg(),
    DirtyDrum(),
    Folding()
], p=1)

# Define your augmentation pipeline here (as in your original code)

defaultPipe = default_augraphy_pipeline()
pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)

# Define the function to process a single image
def process_image(image_file):
    image_path = os.path.join(input_dir, image_file)
    print(image_path)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Apply transformations and augmentations
    for idx in range(10):
        # Slightly shift the image in x or y direction
        shift_x = np.random.randint(-20, 20)  # Random shift in x direction from -5 to 5 pixels
        shift_y = np.random.randint(-20, 20)  # Random shift in y direction from -5 to 5 pixels
        
        # Create the transformation matrix for translation
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        
        # Apply the translation
        transformed_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]), borderValue=(255, 255, 255))

        # Apply the augmentation pipeline to the transformed image
        try:
            augmented_image = pipeline.augment(transformed_image)['output']
        except Exception as e:
            print(f"Error applying pipeline {idx} to image {image_file}: {e}")
            continue

        # Save the dirty and clean images
        dirty_image_path = os.path.join(output_dirty_dir, f'{image_file}_{idx}_aug.png')
        clean_image_path = os.path.join(output_clean_dir, f'clean_{image_file}_{idx}_aug.png')
        
        cv2.imwrite(dirty_image_path, augmented_image)
        cv2.imwrite(clean_image_path, transformed_image)
        print(f'Generated: {dirty_image_path}')

# Use multiprocessing to process images in parallel
if __name__ == '__main__':
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_image, image_files)
