import cv2
import numpy as np
from augraphy import *
import concurrent.futures
import os


# Common receipt dirt and degradation types with their implementations
RECEIPT_DIRT_TYPES = {
    # 1. Coffee stains
    "coffee_stains": AugmentationSequence([
        BleedThrough(intensity_range=(0.3, 0.7),),  # Brown color
        NoiseTexturize(sigma_range=(5, 15))
    ], p=1),
    
    # 2. Water damage
    "water_damage": AugmentationSequence([
        BleedThrough(intensity_range=(0.2, 0.5),),
        # LensFlare(lens_flare_location = "random",
        #               lens_flare_color = "random",
        #               lens_flare_size = (0.5, 5),
        #               ),
        DoubleExposure(gaussian_kernel_range=(2,4),
                offset_direction=1,
                offset_range=(3,6),
                )
    ], p=1),
    
    # 3. Finger smudges
    "finger_smudges": AugmentationSequence([
        InkBleed(intensity_range=(0.1, 0.3),kernel_size=(3, 6)),
        DirtyDrum(line_width_range=(5, 8),line_concentration=0.2)
    ], p=1),
    
    # 4. Crumpled paper
    "crumpled": AugmentationSequence([
        Folding(fold_count=6,fold_noise=2),
        NoiseTexturize(sigma_range=(1,2))
    ], p=1),
    
    # 5. Pocket wear
    "pocket_wear": AugmentationSequence([
        Folding(fold_count=4),
        SubtleNoise(subtle_range=10),
        DirtyDrum()
    ], p=1),
    
    # # 6. Faded ink
    "faded_ink_light": AugmentationSequence([
        Brightness(brightness_range=(1.1, 2.5)),
        Letterpress(value_range=(50, 200)),
    ], p=1),
    # # 6. Faded ink
    "faded_ink_bleed": AugmentationSequence([
        # Brightness(brightness_range=(0.5, 0.9)),
        Letterpress(value_range=(50, 200)),
        InkBleed(intensity_range=(0.2, 0.4),kernel_size=(3, 8)),
    ], p=1),
    
    "hollow": 
    AugmentationSequence([
        Hollow(hollow_median_kernel_value_range = (101, 101),
                hollow_min_width_range=(1, 1),
                hollow_max_width_range=(200, 200),
                hollow_min_height_range=(1, 1),
                hollow_max_height_range=(200, 200),
                hollow_min_area_range=(10, 10),
                hollow_max_area_range=(5000, 5000),
                hollow_dilation_kernel_size_range = (3, 3),
                )
    ], p=1),

    
    # # 7. Oil spots
    "news_paper": AugmentationSequence([
        BleedThrough(intensity_range=(0.1, 0.4),color_range=(0, 224)),
        # NoiseTexturize(sigma_range=(1, 4)),
        InkShifter(text_shift_scale_range=(5, 12))
    ], p=1),
    
    # 8. Dust accumulation
    "dust": AugmentationSequence([
        SubtleNoise(subtle_range=15),
        NoiseTexturize(sigma_range=(2, 5)),
        DirtyDrum(line_concentration=0.15)
    ], p=1),
    
    # 9. Printer defects
    "printer_defects": AugmentationSequence([
        LowInkRandomLines(count_range=(10, 30)),
        InkShifter(text_shift_scale_range=(5, 15))
    ], p=1),
    
    # 10. Age yellowing
    "yellowing": AugmentationSequence([
        ColorShift(color_shift_brightness_range=(0.9,1.1)),
        Brightness(brightness_range=(0.9, 1.1))
    ], p=1),
    
    # 11. Ink bleeding
    "ink_bleeding": AugmentationSequence([
        InkBleed(intensity_range=(0.3, 0.6)),
        BleedThrough(intensity_range=(0.2, 0.4))
    ], p=1),
    
    # 12. Folding marks
    "folding_marks": AugmentationSequence([
        Folding(fold_count=5,
                fold_noise=0.15),
        NoiseTexturize(sigma_range=(1, 3))
    ], p=1),
    
    # 13. Thermal paper fading
    "thermal_fading": AugmentationSequence([
        Brightness(brightness_range=(1.1, 1.4)),
        NoiseTexturize(sigma_range=(1, 3)),
        LightingGradient()
    ], p=1),
    
    # 14. Wrinkles
    "wrinkles": AugmentationSequence([
        Geometric(rotate_range=(0, 10)),
        NoiseTexturize(sigma_range=(2, 5))
    ], p=1),
    
    # 15. Pen marks
    "pen_marks": AugmentationSequence([
        DirtyDrum(line_width_range=(1, 3),
                  line_concentration=0.23),
        InkBleed(intensity_range=(0.1, 0.3))
    ], p=1),
    
    # 16. Humidity damage
    "humidity_damage": AugmentationSequence([
        Squish(),
        BleedThrough(intensity_range=(0.2, 0.4)),
        NoiseTexturize(sigma_range=(3, 8))
    ], p=1),
    
    # 17. Edge tears
    "edge_tears": AugmentationSequence([
        # Geometric(randomize=20),
        DirtyDrum(line_width_range=(2, 5))
    ], p=1),
    
    # # 18. Staple holes
    # "staple_holes": AugmentationSequence([
    #     Geometric(corner_pull_range=(10, 30)),
    #     DirtyDrum(line_width_range=(1, 2),
    #               line_concentration=(0.05, 0.1))
    # ], p=1),
    
    # 19. Chemical stains
    "chemical_stains": AugmentationSequence([
        BleedThrough(intensity_range=(0.2, 0.5),
                    color_range=(150,200)),
        NoiseTexturize(sigma_range=(4, 10))
    ], p=1),
    
    # # 20. Tape residue
    "tape_residue": AugmentationSequence([
        DirtyDrum(line_width_range=(5, 8),
                  line_concentration=0.15),
        NoiseTexturize(sigma_range=(2, 5)),
        Brightness(brightness_range=(0.9, 1.1))
    ], p=1),
       # 1. Basic ink bleeding
    "ink_bleeding_2": AugmentationSequence([
        InkBleed(intensity_range=(0.2, 0.5),
                kernel_size=(5, 10)),
        BleedThrough(intensity_range=(0.2, 0.4),
                    color_range=(0,  250))
    ], p=1),
    
    # 2. Low ink printer effect
    "printer_defects_2": AugmentationSequence([
        LowInkRandomLines(count_range=(50, 200)),
        LowInkPeriodicLines(count_range=(2, 5))
    ], p=1),
    
    # # 3. Dirty printer drum effect
    # "dirty_drum": AugmentationSequence([
    #     DirtyDrum(line_width_range=(1, 3),
    #               line_concentration=(0.1, 0.3)),
    #     DirtyRollers()
    # ], p=1),
    
    # # 4. Paper folding marks
    # "folding": AugmentationSequence([
    #     Folding(fold_count=(2, 4),
    #             fold_noise=(10, 20),
    #             fold_angle_range=(0, 360)),
    #     PageBorder(side="random")
    # ], p=1),
    
    # 5. Lighting issues
    "lighting_problems": AugmentationSequence([
        Brightness(brightness_range=(0.7, 0.9)),
        LightingGradient(
                        max_brightness=255,
                        min_brightness=64,)
    ], p=1),
    
    # 6. Bad photocopy
    "bad_copy": AugmentationSequence([
        BadPhotoCopy(blur_noise=5,
                    noise_sparsity=(0.4, 0.8),
                ),
        NoiseTexturize(sigma_range=(3, 8))
    ], p=1),
    
    # # 7. Letterpress effect
    "letterpress_defects": AugmentationSequence([
        Letterpress(n_samples=(100, 300),
                   n_clusters=(200, 500),
                   std_range=(1000, 3000),
                   value_range=(150, 255)),
        BindingsAndFasteners()
    ], p=1),
    
    # 8. Ink shifting
    "ink_shifting": AugmentationSequence([
        InkShifter( text_shift_scale_range=(18, 27),
                    text_shift_factor_range=(1, 4),
                    text_fade_range=(0, 2),
                    noise_type = "random",),
        SubtleNoise(subtle_range=5)
    ], p=1),
    
    # # 9. Paper warping
    # "warping": AugmentationSequence([
    #     Geometric(scale_range=(0.95, 1.05),
    #              rotation_range=(-5, 5),
    #              translation_range=(-10, 10),
    #              corner_pull_range=(50, 100))
    # ], p=1),
    
    # # 10. Shadow effects
    "shadows": AugmentationSequence([
        ShadowCast(shadow_width_range=(0.3, 0.7),
                   shadow_height_range=(0.5, 0.8),
                  shadow_opacity_range=(0.3, 0.7)),
        Brightness(brightness_range=(0.8, 1.0))
    ], p=1),
    
    # 11. Multiple noise types
    "noise_combination": AugmentationSequence([
        SubtleNoise(subtle_range=10),
        NoiseTexturize(sigma_range=(2, 5)),
        # PaperFactory(texture_path="basic")
    ], p=1),

    # # 16. Edge wear
    "edge_wear": AugmentationSequence([
        PageBorder(),
        DirtyDrum(line_width_range=(1, 2),
                  line_concentration=0.4)
    ], p=1),
    
    # 17. Severe degradation
    "severe_degradation": AugmentationSequence([
       BadPhotoCopy(noise_type=1,
                                   noise_side="left",
                                   noise_iteration=(2,3),
                                   noise_size=(2,3),
                                   noise_sparsity=(0.15,0.15),
                                   noise_concentration=(0.3,0.3),
                                   blur_noise=-1,
                                   blur_noise_kernel=(5, 5),
                                   wave_pattern=0,
                                   edge_effect=0),
        BleedThrough(intensity_range=(0.3, 0.5)),
        Brightness(brightness_range=(0.7, 0.9))
    ], p=1),
    
    # # 18. Color bleeding
    # "color_bleeding": AugmentationSequence([
    #     ColorShift(color_shift_range=(10, 30)),
    #     BleedThrough(intensity_range=(0.2, 0.4)),
    #     LightingGradient()
    # ], p=1),
    
    # 19. Paper texture
    "paper_texture": AugmentationSequence([
    
        NoiseTexturize(sigma_range=(2, 5)),
        SubtleNoise(subtle_range=5)
    ], p=1),
    
    # 20. Mixed deterioration
    "mixed_deterioration": AugmentationSequence([
        InkBleed(intensity_range=(0.1, 0.3)),
        Folding(fold_count=5),
        DirtyDrum(line_width_range=(1, 2)),
        Brightness(brightness_range=(0.8, 1.0))
    ], p=1),
    
}



def create_effects_grid(image_path):
    """
    Creates a grid visualization of all dirt effects applied to an input image
    
    Args:
        image_path: Path to the input image
    
    Returns:
        Grid image showing all effects with labels
    """
    # Read the input image
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    # Calculate grid size (adjust based on number of effects)
    n_cols = 5
    n_rows = 10
    
    # Define padding and text area
    padding = 20
    text_height = 40
    
    # Calculate single cell size (with padding)
    cell_width = original.shape[1] + 2 * padding
    cell_height = original.shape[0] + 2 * padding + text_height

    # Create blank canvas for the grid
    grid_width = cell_width * n_cols
    grid_height = cell_height * n_rows
    grid_image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    
    # Create all effects from the previous dictionary
    effects = RECEIPT_DIRT_TYPES  # Using the dictionary from the previous example
    
    # Process each effect
    for idx, (effect_name, effect_sequence) in enumerate(effects.items()):
        if idx >= n_rows * n_cols:
            print(f"Skipping effect '{effect_name}' as grid is full.")
            break  # Avoid exceeding grid size
        
        # Calculate position in grid
        row = idx // n_cols
        col = idx % n_cols
        
        # Create pipeline for this effect
        pipeline = AugraphyPipeline(
            post_phase=effect_sequence
        )
        
        # Apply effect
        augmented = pipeline.augment(original)
        processed_image = augmented['output']  # Extract the image from the dict
        
        # Ensure processed_image has the same dimensions as original
        if processed_image.shape != original.shape:
            processed_image = cv2.resize(processed_image, (original.shape[1], original.shape[0]))
        
        # Calculate position for this cell
        x_offset = col * cell_width + padding
        y_offset = row * cell_height + padding
        
        # Define the region where the processed image will be placed
        y_end = y_offset + original.shape[0]
        x_end = x_offset + original.shape[1]
        
        # Place processed image
        grid_image[y_offset:y_end, x_offset:x_end] = processed_image
        
        # Add text label
        text_y = y_end + padding + 20  # Adjusted for better positioning
        cv2.putText(grid_image, effect_name,
                    (x_offset, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return grid_image

def add_title(grid_image, title="Receipt Degradation Effects"):
    """
    Adds a title to the top of the grid image
    
    Args:
        grid_image: The grid image to add title to
        title: Title text
    
    Returns:
        Image with title added
    """
    # Create space for title
    title_height = 60
    new_image = np.ones((grid_image.shape[0] + title_height, grid_image.shape[1], 3), dtype=np.uint8) * 255
    
    # Add title
    cv2.putText(new_image, title,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Add grid image below title
    new_image[title_height:, :] = grid_image
    
    return new_image

def generate_effect_comparison(input_image_path, output_path='effect_comparison.jpg'):
    """
    Generate and save a comparison grid of all effects
    
    Args:
        input_image_path: Path to input image
        output_path: Path to save the result
    """
    # Generate grid
    grid = create_effects_grid(input_image_path)
    
    # Add title
    final_image = add_title(grid)
    
    # Save result
    cv2.imwrite(output_path, final_image)
    
    return final_image

# Example usage:

# Generate comparison grid
generate_effect_comparison('./data/clean_base/0001-USPS-dmm300_608.pdf-15_gray.png', 'receipt_effects_comparison.jpg')
