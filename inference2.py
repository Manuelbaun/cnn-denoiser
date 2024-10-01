import numpy as np
import os
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf

IMG_HEIGHT = 400
IMG_WIDTH = 400

def model():
    """trained 80 epochs"""
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

def model6():
    """trained 80 epochs"""
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

    model.load_weights('./checkpoints/autoencoders/checkpoint_6.weights.h5')
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


def model7():
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))

    # encoding
    x = Conv2D(64*3, (4, 4), activation='relu', padding='same')(input_layer)
    x = Conv2D(128*3, (4, 4), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Dropout(0.4)(x)

    # decoding
    x = Conv2D(128*3, (4, 4), activation='relu', padding='same')(x)
    x = Conv2D(64*3, (4, 4), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)

    output_layer = Conv2D(1, (4, 4), activation='sigmoid', padding='same')(x)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer='adam' , loss='mean_squared_error', metrics=['mae'])

    model.load_weights('./checkpoints/autoencoders/checkpoint_7.weights.h5')
    model.summary()
    return model




def split_image(image):
    patches = []
    img_height, img_width = image.shape  # (height, width)
    patch_height, patch_width = IMG_HEIGHT, IMG_WIDTH  # (height, width)

    row_index = 0
    for top in range(0, img_height, patch_height):
        bottom = min(top + patch_height, img_height)
        col_index = 0
        for left in range(0, img_width, patch_width):
            right = min(left + patch_width, img_width)
            patch = image[top:bottom, left:right]

            # Pad patch if it's smaller than patch size
            if patch.shape[0] < patch_height or patch.shape[1] < patch_width:
                padded_patch = np.zeros((patch_height, patch_width), dtype=np.float32)
                padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded_patch
            else:
                patch = patch.astype(np.float32)

            patches.append((patch, (row_index, col_index)))
            col_index += 1
        row_index += 1

    return patches

def reassemble_image(patches, image_size):
    patch_height, patch_width = patches[0][0].shape  # (height, width)

    max_row = max(position[0] for _, position in patches)
    max_col = max(position[1] for _, position in patches)

    reconstructed_height = (max_row + 1) * patch_height
    reconstructed_width = (max_col + 1) * patch_width

    reconstructed_image = np.zeros((reconstructed_height, reconstructed_width), dtype=np.float32)

    for patch, (row, col) in patches:
        reconstructed_image[row * patch_height: (row + 1) * patch_height,
                            col * patch_width: (col + 1) * patch_width] = patch

    # Crop the reconstructed image to the size before scaling
    image_height, image_width = image_size  # (height, width)
    reconstructed_image = reconstructed_image[:image_height, :image_width]

    return reconstructed_image.astype(np.uint8)

# Create output directory if it doesn't exist
output_dir = './denoised_images'
os.makedirs(output_dir, exist_ok=True)

# Path to the test_invoice folder
input_dir = './data/test_invoice'


print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
# Configure TensorFlow to use the GPU and set memory growth
# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to prevent TensorFlow from allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU.")

print("------------------------------------")
# Initialize the model
model = model5()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# For full integer quantization
# converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
tflite_model.summary()

exit(0)
# Batch size configuration
batch_size = 1

# Get the list of image files
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# Process images in batches
for i in range(0, len(image_files), batch_size):
    batch_files = image_files[i:i+batch_size]
    print(f"\nProcessing batch {i // batch_size + 1} with {len(batch_files)} images")

    # Initialize lists to hold data
    images = []
    image_names = []
    original_sizes = []
    scaled_sizes = []
    all_patches = []
    patch_info = []  # To store info about each patch: image index, position

    # Load and process each image in the batch
    for idx, filename in enumerate(batch_files):
        input_path = os.path.join(input_dir, filename)
        print(f"Loading {input_path}...")

        # Load image using OpenCV
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image {input_path}")
            continue

        # Save original image size
        orig_height, orig_width = image.shape  # (height, width)
        original_sizes.append((orig_height, orig_width))
        image_names.append(filename)

        # Define scaling factors
        fx = 1.2 if orig_width <= 500 else 0.8  # Width is image.shape[1]
        fy = 1.2 if orig_width <= 500 else 0.8  # Width is image.shape[1]

        # Calculate new dimensions
        new_width = int(orig_width * fx)
        new_height = int(orig_height * fy)
        scaled_sizes.append((new_height, new_width))

        # Resize the image using INTER_CUBIC interpolation
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Split the image into patches
        patches = split_image(image)

        # Collect all patches and their info
        for patch_array, position in patches:
            all_patches.append(patch_array)
            patch_info.append({'image_index': idx, 'position': position})

    if not all_patches:
        print("No patches to process in this batch.")
        continue

    # Prepare patches for the model
    prepared_patches = [patch.reshape(1, patch.shape[0], patch.shape[1], 1) / 255.0 for patch in all_patches]
    prepared_patches_array = np.concatenate(prepared_patches, axis=0)

    # Process all patches with your model
    denoised_patches_output = model.predict(prepared_patches_array)
    # denoised_patches_output = model.predict(denoised_patches_output)

    # Post-process the outputs
    denoised_patches_per_image = {idx: [] for idx in range(len(batch_files))}

    for denoised_patch, info in zip(denoised_patches_output, patch_info):
        image_index = info['image_index']
        position = info['position']

        denoised_patch = denoised_patch.squeeze() * 255.0  # Denormalize
        denoised_patch = denoised_patch.astype(np.float32)
        denoised_patches_per_image[image_index].append((denoised_patch, position))

    # Reassemble and save images
    for idx in range(len(batch_files)):
        patches = denoised_patches_per_image[idx]
        if not patches:
            print(f"No patches found for image {image_names[idx]}")
            continue

        # Reassemble the processed patches into an image
        reconstructed_image_array = reassemble_image(patches, scaled_sizes[idx])

        # Resize the image back to the original size
        reconstructed_image_array = cv2.resize(
            reconstructed_image_array,
            (original_sizes[idx][1], original_sizes[idx][0]),
            interpolation=cv2.INTER_CUBIC
        )

        # Save the reconstructed image
        output_path = os.path.join(output_dir, f"denoised_{image_names[idx]}")
        cv2.imwrite(output_path, reconstructed_image_array)
        print(f"Saved denoised image to {output_path}")