import numpy as np
from PIL import Image

def apply_convolution(image_array, kernel):
    """
    Applies a convolution operation to a 2D grayscale image array.

    Args:
        image_array (np.array): A 2D NumPy array representing the grayscale image.
        kernel (np.array): A 2D NumPy array representing the convolution kernel.

    Returns:
        np.array: The convolved (processed) image array.
    """
    # Get image and kernel dimensions
    image_height, image_width = image_array.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate padding needed. For a 3x3 kernel, pad by 1 on each side.
    # This ensures the output image has the same dimensions as the input.
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    # Create a padded version of the image to handle borders during convolution.
    # We use 'constant' padding with a value of 0, meaning pixels outside the
    # image boundaries are treated as 0. Other modes like 'edge' (replicating
    # border pixels) can also be used.
    padded_image = np.pad(image_array, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # Initialize the output image array with zeros, same size as input image
    output_image = np.zeros_like(image_array, dtype=float)

    # Iterate over each pixel in the output image
    for y in range(image_height):
        for x in range(image_width):
            # Extract the region of the padded image corresponding to the kernel's position
            # The region is centered at (y, x) in the original image.
            region = padded_image[y:y + kernel_height, x:x + kernel_width]

            # Perform element-wise multiplication of the region and the kernel,
            # then sum the results. This is the convolution operation.
            output_image[y, x] = np.sum(region * kernel)

    # Ensure output pixel values are within a valid image range (0-255 for grayscale)
    # and convert to integer type for displaying.
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    return output_image

# --- 1. Create a sample grayscale image ---
# A simple 8x8 image to clearly show pixel values changing
# It has distinct blocks of values to make blurring visible.
original_image_array = np.array([
    [50, 50, 50, 50, 200, 200, 200, 200],
    [50, 50, 50, 50, 200, 200, 200, 200],
    [50, 50, 50, 50, 200, 200, 200, 200],
    [50, 50, 50, 50, 200, 200, 200, 200],
    [100, 100, 100, 100, 150, 150, 150, 150],
    [100, 100, 100, 100, 150, 150, 150, 150],
    [100, 100, 100, 100, 150, 150, 150, 150],
    [100, 100, 100, 100, 150, 150, 150, 150]
], dtype=np.uint8)

print("Original Image Array (8x8 pixels):")
print(original_image_array)

# --- 2. Define a 3x3 blurring kernel (averaging filter) ---
# Each element is 1/9, so it averages the 9 pixels under the kernel.
blur_kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=float) / 9

print("\nBlurring Kernel (3x3):")
print(blur_kernel)

# --- 3. Apply the convolution operation ---
blurred_image_array = apply_convolution(original_image_array, blur_kernel)

print("\nBlurred Image Array (8x8 pixels):")
print(blurred_image_array)

# --- 4. Visualize the original and blurred images ---
# Convert NumPy arrays to PIL Image objects
original_image = Image.fromarray(original_image_array, mode='L') # 'L' for grayscale
blurred_image = Image.fromarray(blurred_image_array, mode='L')

# Enlarge images for better visualization (e.g., 400x400 pixels for an 8x8 image)
# Using Image.NEAREST for original to keep crisp pixels, and Image.BILINEAR for blurred
# to show the smoothing.
original_image_display = original_image.resize((400, 400), Image.NEAREST)
blurred_image_display = blurred_image.resize((400, 400), Image.BILINEAR)

# Display the images (this will open image viewer windows if run locally)
# If running in an environment without a GUI, these lines might not work directly.
# You can save them instead:
# original_image_display.save("original_image.png")
# blurred_image_display.save("blurred_image.png")

print("\nDisplaying original and blurred images. Look for pop-up windows.")
print("If no windows appear, the images have been processed and their arrays printed above.")
print("You might need to save them to view them, or run this code in an environment with GUI support.")

try:
    original_image_display.show(title="Original Image")
    blurred_image_display.show(title="Blurred Image")
except Exception as e:
    print(f"\nCould not display images (often due to missing GUI environment): {e}")
    print("Images can be saved by uncommenting the .save() lines in the code.")