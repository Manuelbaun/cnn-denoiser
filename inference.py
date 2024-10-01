import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2 # import the l2 regularizer from keras.regularizers


IMG_HEIGHT=400
IMG_WIDTH=400


def model():
    """trained 80 epochs"""
    IMG_HEIGHT=400
    IMG_WIDTH=400

    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    
    # encoding
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Dropout(0.5)(x)

    # decoding
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)

    output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer='adam' , loss='mean_squared_error', metrics=['mae'])

    model.load_weights('./checkpoints/autoencoders/checkpoint_1.weights.h5')
    model.summary()
    return model


def model4():
    """trained 150 epochs"""
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))

    # encoding
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Dropout(0.5)(x)

    # decoding
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)

    output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer='adam' , loss='mean_squared_error', metrics=['mae'])
    model.load_weights('./checkpoints/autoencoders/checkpoint_4.weights.h5')
    model.summary()
    return model


def model2():
    """trained 80 epochs"""
    IMG_HEIGHT=400
    IMG_WIDTH=400
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))

    # encoding with more layers and filters
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)

    # decoding
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)

    output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])

    model.load_weights('./checkpoints/autoencoders/checkpoint_2.weights.h5')
    model.summary()
    return model

def model3():
    """trained 150 epochs"""
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))

    # encoding with more layers and filters
    x = Conv2D(100, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(200, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    x = Conv2D(400, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)

    # decoding
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(200, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(100, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)

    output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])

    model.load_weights('./checkpoints/autoencoders/checkpoint_3.weights.h5')
    model.summary()
    return model

def model5():
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))

    # encoding
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Dropout(0.5)(x)

    # decoding
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)

    output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer='adam' , loss='mean_squared_error', metrics=['mae'])

    model.load_weights('./checkpoints/autoencoders/checkpoint_5.weights.h5')
    model.summary()
    return model




# model = optimized_model()

# model = model()
model = model()

def split_image(image):
    patches = []
    img_width, img_height = image.size  # (width, height)
    patch_width, patch_height = IMG_WIDTH, IMG_HEIGHT  # (width, height)

    row_index = 0
    for top in range(0, img_height, patch_height):
        bottom = min(top + patch_height, img_height)
        col_index = 0
        for left in range(0, img_width, patch_width):
            right = min(left + patch_width, img_width)
            box = (left, top, right, bottom)
            patch = image.crop(box)
            patch_array = np.asarray(patch, dtype=np.float32)

            # Pad patch if it's smaller than patch size
            if patch_array.shape[0] < patch_height or patch_array.shape[1] < patch_width:
                padded_patch = np.zeros((patch_height, patch_width), dtype=np.float32)
                padded_patch[:patch_array.shape[0], :patch_array.shape[1]] = patch_array
                patch_array = padded_patch

            patches.append((patch_array, (row_index, col_index)))
            col_index += 1
        row_index += 1

    return patches

def reassemble_image(patches, original_size):
    patch_height, patch_width = patches[0][0].shape  # (height, width)

    max_row = max(position[0] for _, position in patches)
    max_col = max(position[1] for _, position in patches)

    reconstructed_height = (max_row + 1) * patch_height
    reconstructed_width = (max_col + 1) * patch_width

    reconstructed_image = np.zeros((reconstructed_height, reconstructed_width), dtype=np.float32)

    for patch, (row, col) in patches:
        reconstructed_image[row * patch_height: (row + 1) * patch_height,
                            col * patch_width: (col + 1) * patch_width] = patch

    # Crop the reconstructed image to the original size
    orig_width, orig_height = original_size  # (width, height)
    reconstructed_image = reconstructed_image[:orig_height, :orig_width]

    return reconstructed_image.astype(np.uint8)



# Create output directory if it doesn't exist
output_dir = './denoised_images'
os.makedirs(output_dir, exist_ok=True)

# Path to the test_invoice folder
input_dir = './data/test_invoice'

# Iterate over all image files in the folder
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_path = os.path.join(input_dir, filename)
        print(f"Processing {input_path}...")

        # Load the image
        image = Image.open(input_path).convert('L')  # Convert to grayscale if needed

        # Save original image size
        original_size = image.size  # (width, height)
        # Define scaling factors
        fx = 2 if original_size[0] <= 400 else 0.75  # Scaling factor along the x-axis
        fy = 0.75  # Scaling factor along the y-axis

        # Calculate new dimensions
        new_width = int(original_size[0] * fx)
        new_height = int(original_size[1] * fy)

        # Resize the image using BICUBIC interpolation
        image = image.resize((new_width, new_height), Image.BICUBIC)



        # Split the image into patches
        patches = split_image(image)



        # ---------------
        # Prepare patches for the model
        prepared_patches = []
        positions = []
        for patch_array, position in patches:
            # Prepare the patch for the model
            patch_input = patch_array.reshape(1, patch_array.shape[0], patch_array.shape[1], 1) / 255.0
            prepared_patches.append(patch_input)
            positions.append(position)

        # Convert the list of prepared patches into a numpy array for batch processing
        prepared_patches_array = np.concatenate(prepared_patches, axis=0)

        # Process all patches with your model
        denoised_patches_output = model.predict(prepared_patches_array)

        # Post-process the outputs
        denoised_patches = []
        for denoised_patch, position in zip(denoised_patches_output, positions):
            denoised_patch = denoised_patch.squeeze() * 255.0  # Denormalize
            denoised_patch = denoised_patch.astype(np.float32)
            denoised_patches.append((denoised_patch, position))

        # ---------------

        # Reassemble the processed patches into an image
        reconstructed_image_array = reassemble_image(denoised_patches, original_size)

        # Convert the NumPy array back to a PIL image
        reconstructed_image = Image.fromarray(reconstructed_image_array)

        # Save the reconstructed image
        output_path = os.path.join(output_dir, f"denoised_{filename}")
        reconstructed_image.save(output_path)
