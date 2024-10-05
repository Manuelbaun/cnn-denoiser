from multiprocessing import Pool, cpu_count
import cv2
import os
import numpy as np
from augraphy import *



# Directory paths
input_dir = 'data/clean_base'        # Replace with your actual path
output_clean_dir = 'data/train_cleaned' # Replace with your actual path
output_dirty_dir = 'data/train' # Replace with your actual path
output_test_dir = 'data/test' # Replace with your actual path
output_debug_dir = 'data/debug' # Replace with your actual path

# Create output directories if they don't exist
os.makedirs(output_clean_dir, exist_ok=True)
os.makedirs(output_dirty_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)
os.makedirs(output_debug_dir, exist_ok=True)

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
    Letterpress(n_samples=(200, 500),
                n_clusters=(300, 800),
                std_range=(1500, 5000),
                value_range=(200, 255),
                value_threshold_range=(100, 128),
                blur=1),
    InkShifter(
                text_shift_scale_range=(18, 27),
                text_shift_factor_range=(1, 4),
                text_fade_range=(0, 2),
                noise_type = "random",)
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
    # print(image_path)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Apply transformations and augmentations
    for idx in range(15):

        try:        
            augmented_image = defaultPipe.augment(image)['output']
        except Exception as e:
            print(f"Error applying pipeline {idx} to image {image_file}: {e}")
            continue


        clean_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
        dirty_image = augmented_image
        # Check if augmented_image is valid
        if dirty_image is None:
            print(f"Augmented image is invalid for {image_file} at index {idx}. Skipping...")
            continue
        
        
        if len(augmented_image.shape) > 2:
            print(f"Augmented image is not gray {image_file} shape: {len(augmented_image.shape)}: convert")
            try:
                dirty_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2GRAY)    
            except Exception as e:
                print(f"Error applying color convert {idx} to image {image_file}: {e}")
                continue


 
        # Ensure both images have the same height
        # if clean_image.shape[0] != dirty_image.shape[0]:
        #     height = max(clean_image.shape[0], dirty_image.shape[0])  # Choose the max height
        #     clean_image = cv2.resize(clean_image, (int(clean_image.shape[1] * (height / clean_image.shape[0])), height))
        #     dirty_image = cv2.resize(dirty_image, (int(dirty_image.shape[1] * (height / dirty_image.shape[0])), height))

        # Combine images horizontally
        # combined_image = np.hstack((clean_image, dirty_image))
        # # Save the combined image
        # combined_image_path = os.path.join(output_debug_dir, f'{image_file}_{idx}_debug.png')
        # cv2.imwrite(combined_image_path, combined_image)
        
        # # print(f'Generated: {combined_image_path}')
        # continue

        # Save the dirty and clean images
        dirty_image_path = os.path.join(output_dirty_dir, f'{image_file}_{idx}_aug.png')
        clean_image_path = os.path.join(output_clean_dir, f'clean_{image_file}_{idx}_aug.png')
        # test_image_path = os.path.join(output_test_dir, f'{image_file}_{idx}test.png')
        
        #cv2.imwrite(test_image_path, augmented_image)
        cv2.imwrite(dirty_image_path, dirty_image)
        cv2.imwrite(clean_image_path, clean_image)
        print(f'Generated: {dirty_image_path}')

# Use multiprocessing to process images in parallel
if __name__ == '__main__':
    with Pool(processes=cpu_count()-2) as pool:
        pool.map(process_image, image_files)


