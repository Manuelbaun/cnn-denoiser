from augraphy import *

# Pipeline 1: Perspective and Geometric Distortion
perspective_pipeline = AugraphyPipeline(
    ink_phase=AugmentationSequence([
        Brightness(brightness_range=(0.8, 1.2)),
        InkBleed(intensity_range=(0.1, 0.3))
    ], p=1),
    paper_phase=AugmentationSequence([
        PaperFactory(),
        Geometric(scale_range=(0.85, 1.15),
                 rotation_range=(-15, 15),
                 translation_range=(-50, 50),
                 corner_pull_range=(50, 200))
    ], p=1)
)

# Pipeline 2: Poor Lighting Conditions
lighting_pipeline = AugraphyPipeline(
    ink_phase=AugmentationSequence([
        Brightness(brightness_range=(0.4, 0.8)),
        LightingGradient(light_position="random",
                        direction="random",
                        max_brightness=255,
                        min_brightness=64),
        ShadowCast(shadow_size_range=(0.3, 0.7),
                   shadow_opacity_range=(0.3, 0.7))
    ], p=1),
    paper_phase=AugmentationSequence([
        PaperFactory(texture_path="basic")
    ], p=1)
)

# Pipeline 3: Bad Photo Quality
bad_photo_pipeline = AugraphyPipeline(
    ink_phase=AugmentationSequence([
        Brightness(brightness_range=(0.6, 1.1)),
        BleedThrough()
    ], p=1),
    paper_phase=AugmentationSequence([
        BadPhotoCopy(blur_noise=2,
                    noise_intensity=(0.3, 0.7),
                    blur_range=(1, 3)),
        NoiseTexturize(sigma_range=(3, 10))
    ], p=1),
    post_phase=AugmentationSequence([
        Jpeg(quality_range=(30, 70))
    ], p=1)
)

# Pipeline 4: Receipt Bends and Folds
bend_pipeline = AugraphyPipeline(
    paper_phase=AugmentationSequence([
        PaperFactory(),
        Folding(fold_count=(2, 4),
                fold_noise=(10, 20),
                fold_angle_range=(0, 360)),
        Geometric(corner_pull_range=(20, 100))
    ], p=1)
)

# Pipeline 5: Ink Bleeding and Degradation
ink_degradation_pipeline = AugraphyPipeline(
    ink_phase=AugmentationSequence([
        InkBleed(intensity_range=(0.3, 0.7),
                kernel_size=(5, 10)),
        BleedThrough(intensity_range=(0.2, 0.5)),
        LowInkRandomLines(count_range=(50, 200))
    ], p=1),
    paper_phase=AugmentationSequence([
        PaperFactory(texture_path="basic")
    ], p=1)
)

# Pipeline 6: Unclear Letters and Text
unclear_text_pipeline = AugraphyPipeline(
    ink_phase=AugmentationSequence([
        Letterpress(n_samples=(100, 300),
                   n_clusters=(200, 400),
                   std_range=(1000, 3000)),
        InkShifter(text_shift_scale_range=(5, 15),
                   text_shift_factor_range=(1, 3)),
        LowInkLine()
    ], p=1)
)

# Pipeline 7: Heavy Shadows
shadow_pipeline = AugraphyPipeline(
    ink_phase=AugmentationSequence([
        ShadowCast(shadow_size_range=(0.5, 0.9),
                   shadow_opacity_range=(0.4, 0.8),
                   number_of_shadows=(2, 4)),
        LightingGradient(light_position="random",
                        direction="random")
    ], p=1),
    paper_phase=AugmentationSequence([
        PaperFactory()
    ], p=1)
)

# Pipeline 8: Ink Shifting and Bleeding
ink_shift_pipeline = AugraphyPipeline(
    ink_phase=AugmentationSequence([
        InkShifter(text_shift_scale_range=(10, 30),
                   text_shift_factor_range=(2, 5),
                   text_fade_range=(1, 3)),
        InkBleed(intensity_range=(0.2, 0.6)),
        BleedThrough()
    ], p=1)
)

# Pipeline 9: Heavy Folding and Creases
folding_pipeline = AugraphyPipeline(
    paper_phase=AugmentationSequence([
        Folding(fold_count=(3, 6),
                fold_noise=(15, 25),
                fold_angle_range=(0, 360)),
        Geometric(corner_pull_range=(30, 150)),
        NoiseTexturize(sigma_range=(2, 8))
    ], p=1)
)

# Pipeline 10: Mixed Noise and Dirt
dirty_pipeline = AugraphyPipeline(
    ink_phase=AugmentationSequence([
        BleedThrough(),
        InkBleed()
    ], p=1),
    paper_phase=AugmentationSequence([
        PaperFactory(),
        NoiseTexturize(sigma_range=(2, 8)),
        SubtleNoise(subtle_range=10),
        DirtyDrum(line_width_range=(1, 3),
                  line_concentration=(0.1, 0.3))
    ], p=1),
    post_phase=AugmentationSequence([
        Jpeg(quality_range=(50, 90)),
        NoiseTexturize()
    ], p=1)
)

# List of all pipelines for easy access
pipelines = [
    perspective_pipeline,
    lighting_pipeline,
    bad_photo_pipeline,
    bend_pipeline,
    ink_degradation_pipeline,
    unclear_text_pipeline,
    shadow_pipeline,
    ink_shift_pipeline,
    folding_pipeline,
    dirty_pipeline
]