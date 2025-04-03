from PIL import Image
import numpy as np

def load_image(path: str, width: int, height: int):
  """
  Reads an image in grayscale, downscales it while maintaining aspect ratio,
  and pads it to fit the specified dimensions.

  Args:
    path: The path to the image file.
    width: The target width.
    height: The target height.

  Returns:
    A NumPy array representing the processed grayscale image.
  """
  img = Image.open(path).convert('L')
  scale_down_factor = min(width / img.width, height / img.height)
  new_width = int(img.width * scale_down_factor)
  new_height = int(img.height * scale_down_factor)
  img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

  padded_image = np.zeros((height, width), dtype=np.uint8)

  offset_x = (width - new_width) // 2
  offset_y = (height - new_height) // 2

  padded_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = 255 - np.array(img)

  return padded_image

def print_image(img: np.ndarray, ascii_chars: list[str] = [" ", ".", ",", "-", "_", ":", ";", "o", "O", "#", "@"]):
  """
  Prints an image in grayscale, using the specified ASCII characters.

  Args:
    img: The image to print.
    ascii_chars: The ASCII characters to use for each pixel value.
  """
  num_chars = len(ascii_chars)
  min_val = np.min(img)
  max_val = np.max(img)
  diff = max_val - min_val
  for row in img:
    ascii_row = ""
    for pixel_value in row:
      # Map pixel value (0-255) to the index of the ASCII characters
      index = int((pixel_value - min_val) / diff * (num_chars - 1))
      ascii_row += ascii_chars[index]
    print(ascii_row)

def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
  """
  Normalizes a convolution kernel by dividing each element by the sum of all elements.

  Args:
    kernel: A 2D NumPy array representing the convolution kernel.

  Returns:
    A 2D NumPy array representing the normalized convolution kernel.
  """
  if np.min(kernel) < 0: raise ValueError("Kernel with negative values cannot be normalized.")
  return kernel / np.sum(kernel)


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
  """
  Performs 2D convolution of an image with a kernel.

  Args:
    image: A 2D NumPy array representing the input image.
    kernel: A 2D NumPy array representing the convolution kernel.

  Returns:
    A 2D NumPy array representing the convolved image.
  """
  image_height, image_width = image.shape
  kernel_height, kernel_width = kernel.shape
  padding_height, padding_width = kernel_height // 2, kernel_width // 2

  padded_image = np.pad(image, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant')
  output_image = np.zeros(image.shape, dtype=image.dtype)

  for y in range(image_height):
    for x in range(image_width):
      region = padded_image[y:y + kernel_height, x:x + kernel_width]
      output_image[y, x] = np.sum(region * kernel)

  return output_image

def _test_load_image():
  """
  Tests the load_image function.
  """
  img = load_image('../dataset/0/Zero_full (1).jpg', 50, 50)
  print_image(img)

def _test_convolve():
  """
  Tests the convolve function.
  """
  img = load_image('../dataset/0/Zero_full (1).jpg', 50, 50)

  print("Blurred image")
  print_image(
    convolve(img, normalize_kernel(np.array([
      [1, 1, 1, 1],
      [1, 1, 1, 1],
      [1, 1, 1, 1],
      [1, 1, 1, 1],
    ])))
  )

  print("Horizontal Edge Detection")
  print_image(
    convolve(img, np.array([
      [-1,-2,-1],
      [ 0, 0, 0],
      [ 1, 2, 1],
    ]))
  )

  print("Vertical Edge Detection")
  print_image(
    convolve(img, np.array([
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1],
    ]))
  )

if __name__ == '__main__':
  _test_load_image()
  _test_convolve()

