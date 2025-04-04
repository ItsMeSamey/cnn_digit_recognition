from PIL import Image
import numpy as np
import math

from utils.cnn import ImageIterator

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

def convolve(image: np.ndarray, kernel: np.ndarray, stride: tuple[int, int] = (1, 1)) -> np.ndarray:
  """
  Performs 2D convolution of an image with a kernel, respecting stride.

  Args:
    image: A 2D NumPy array representing the input image.
    kernel: A 2D NumPy array representing the convolution kernel.
    stride: A tuple with two integers representing the stride (stride_x, stride_y) of the convolution.

  Returns:
    A 2D NumPy array representing the convolved image.
  """
  image_height, image_width = image.shape
  kernel_height, kernel_width = kernel.shape
  stride_x, stride_y = stride

  output_height = math.ceil(image_height / stride_y)
  if output_height * stride_y > image_height: output_height -= 1
  output_width = math.ceil(image_width / stride_x)
  if output_width * stride_x > image_width: output_width -= 1
  output_image = np.zeros((output_height, output_width), dtype=image.dtype)

  padding_height = (kernel_height // 2, output_height * stride_y + kernel_height - image_height - kernel_height // 2)
  padding_width = (kernel_width // 2, output_width * stride_x + kernel_width - image_width - kernel_width // 2)
  padded_image = np.pad(image, (padding_height, padding_width), mode='constant', constant_values=0)

  for out_y in range(output_height):
    for out_x in range(output_width):
      in_y = out_y * stride_y
      in_x = out_x * stride_x
      region = padded_image[in_y : in_y + kernel_height, in_x : in_x + kernel_width]
      output_image[out_y, out_x] = np.sum(region * kernel)

  print(output_image.shape)
  return output_image

def normalize_kernel_rc(kernel_xy: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
  """
  Normalizes a convolution kernel by dividing each element by the sum of all elements.

  Args:
    kernel_xy: A tuple containing two 1D NumPy arrays (kernel_x, kernel_y) representing the row and column convolution kernels.

  Returns:
    A 2D NumPy array representing the normalized convolution kernel.
  """
  kernel_x, kernel_y = kernel_xy
  if np.min(kernel_x[0]) < 0 or np.min(kernel_y[0]) < 0: raise ValueError("Kernel with negative values cannot be normalized.")
  sum = math.sqrt(np.sum(kernel_x * kernel_y.reshape(-1, 1)))
  return (kernel_x / sum, kernel_y / sum)

def convolve_rc(image: np.ndarray, kernel_xy: tuple[np.ndarray, np.ndarray], stride: tuple[int, int] = (1, 1)) -> np.ndarray:
  """
  Performs 2D separable convolution using a row and column kernel, respecting stride.

  Convolution is performed first along rows (with kernel_x and stride_x),
  then along columns (with kernel_y and stride_y).

  Args:
    image: A 2D NumPy array representing the input image.
    kernel_xy: A tuple containing two 1D NumPy arrays (kernel_x, kernel_y) representing the row and column convolution kernels.
    stride: A tuple with two integers representing the stride (stride_x, stride_y) of the convolution.

  Returns:
    A 2D NumPy array representing the convolved image.
  """
  kernel_x, kernel_y = kernel_xy

  if kernel_y.ndim != 1 or kernel_x.ndim != 1:
    raise ValueError("Kernels in kernel_xy must be 1D arrays.")

  image_height, image_width = image.shape
  kernel_height, kernel_width = kernel_y.shape[0], kernel_x.shape[0]
  stride_x, stride_y = stride

  output_height = math.ceil(image_height / stride_y)
  if output_height * stride_y > image_height: output_height -= 1
  output_width = math.ceil(image_width / stride_x)
  if output_width * stride_x > image_width: output_width -= 1
  output_image = np.zeros((output_height, output_width), dtype=image.dtype)

  padding_height = (kernel_height // 2, output_height * stride_y + kernel_height - image_height - kernel_height // 2)
  padding_width = (kernel_width // 2, output_width * stride_x + kernel_width - image_width - kernel_width // 2)
  padded_image = np.pad(image, ((0, 0), padding_width), mode='constant', constant_values=0)

  intermediate_image = np.zeros((image_height + padding_height[0] + padding_height[1], output_width), dtype=image.dtype)

  for y in range(image_height): # Iterate through each row
    for out_x in range(output_width): # Iterate through output columns
      in_x = out_x * stride_x
      region = padded_image[y: y + 1, in_x: in_x + kernel_width]
      intermediate_image[y + padding_height[0], out_x] = np.sum(region * kernel_x)

  kernel_y = kernel_y.reshape(-1, 1)
  print(padding_width, padding_height)
  for x in range (output_width):
    for out_y in range(output_height):
      in_y = out_y * stride_y
      region = intermediate_image[in_y: in_y + kernel_height, x: x + 1]
      output_image[out_y, x] = np.sum(region * kernel_y)

  return output_image

def _test_load_image():
  """
  Tests the load_image function.
  """
  img = load_image('../datasets/panghalvishesh handwritten-digit/0/Zero_full (1).jpg', 50, 50)
  print_image(img)

def _test_convolve():
  """
  Tests the convolve function.
  """
  img = load_image('../datasets/panghalvishesh handwritten-digit/0/Zero_full (1).jpg', 50, 50)

  print("Blurred image")
  print_image(
    convolve(img, normalize_kernel(np.array([
      [1, 2, 1],
      [2, 4, 2],
      [1, 2, 1],
    ])))
  )

  print("Blurred image rc")
  print_image(
    convolve_rc(img, normalize_kernel_rc((
      np.array([1, 2, 1]),
      np.array([1, 2, 1])
    )))
  )

if __name__ == '__main__':
  _test_load_image()
  _test_convolve()

import gzip
import struct

def load_mnist_images(filename):
  """Loads the MNIST images from a gzipped file using struct."""
  with gzip.open(filename, 'rb') as f:
    magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
    if magic != 2051: raise ValueError('Magic number mismatch in image file: expected 2051, got {}'.format(magic))
    images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
  return images

def load_mnist_labels(filename):
  """Loads the MNIST labels from a gzipped file using struct."""
  with gzip.open(filename, 'rb') as f:
    magic, num = struct.unpack(">II", f.read(8))
    if magic != 2049:
      raise ValueError('Magic number mismatch in label file: expected 2049, got {}'.format(magic))
    labels = np.frombuffer(f.read(), dtype=np.uint8)
  return labels

def _test_load_mnist_images():
  """
  Tests the load_mnist_images function.
  """
  images = load_mnist_images('../datasets/mnist/train-images.idx3-ubyte.gz')
  labels = load_mnist_labels('../datasets/mnist/train-labels.idx1-ubyte.gz')
  print(images.shape)
  print(labels.shape)

  print("Image of", labels[0])
  print_image(images[0])

if __name__ == '__main__':
  _test_load_mnist_images()

class MNISTIterator(ImageIterator):
  def __init__(self, images: np.ndarray, labels: np.ndarray):
    self.images = images
    self.labels = labels
    self.index = 0

  def has_next(self) -> bool:
    return self.index < len(self.images)

  def next(self) -> tuple[np.ndarray, int]:
    retval = self.images[self.index], self.labels[self.index]
    self.index += 1
    return retval

def MNISTTrainIterator() -> ImageIterator:
  """
  Returns an iterator over the training set of the MNIST dataset.

  Returns:
    An iterator over the training set of the MNIST dataset.
  """
  return MNISTIterator(load_mnist_images('../datasets/mnist/train-images.idx3-ubyte.gz'), load_mnist_labels('../datasets/mnist/train-labels.idx1-ubyte.gz'))

def MNISTTestIterator() -> ImageIterator:
  """
  Returns an iterator over the test set of the MNIST dataset.

  Returns:
    An iterator over the test set of the MNIST dataset.
  """
  return MNISTIterator(load_mnist_images('../datasets/mnist/t10k-images.idx3-ubyte.gz'), load_mnist_labels('../datasets/mnist/t10k-labels.idx1-ubyte.gz'))

