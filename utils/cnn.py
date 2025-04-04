from abc import ABCMeta, abstractmethod
from typing import Callable
import numpy as np
import pickle

from utils.functions import softmax

class ConvolutionalLayer(metaclass=ABCMeta):
  @abstractmethod
  def convolve(self, image: np.ndarray) -> np.ndarray:
    raise NotImplementedError

  @abstractmethod
  # should randomly initialize the kernel
  def reset_kernel(self):
    raise NotImplementedError

  # @abstractmethod
  # # returns the kernel of this layer
  # def get_kernel(self) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
  #   raise NotImplementedError
  #
  # @abstractmethod
  # # set the kernel to the given value
  # def set_kernel(self, kernel: np.ndarray):
  #   raise NotImplementedError

class FullyConnectedLayer(metaclass=ABCMeta):
  @abstractmethod
  def activate(self, data: np.ndarray) -> np.ndarray:
    raise NotImplementedError

  @abstractmethod
  # should randomly initialize the weights
  def reset_weights(self):
    raise NotImplementedError

  # @abstractmethod
  # # returns the weights of this layer
  # def get_weights(self) -> np.ndarray:
  #   raise NotImplementedError
  #
  # @abstractmethod
  # # set the weights to the given value
  # def set_weights(self, weights: np.ndarray):
  #   raise NotImplementedError

class ImageIterator(metaclass=ABCMeta):
  @abstractmethod
  def has_next(self) -> bool:
    raise NotImplementedError

  @abstractmethod
  def next(self) -> tuple[np.ndarray, int]:
    raise NotImplementedError

class CNN():
  def __init__(self, convolution_layers: list[ConvolutionalLayer], neural_net: list[FullyConnectedLayer]):
    self.convolution_layers = convolution_layers
    self.neural_net = neural_net

  def store(self) -> bytes:
    return pickle.dumps(self)

  @classmethod
  def load(cls, data: bytes) -> 'CNN':
    return pickle.loads(data)

  def forward(self, image: np.ndarray) -> np.ndarray:
    for layer in self.convolution_layers: image = layer.convolve(image)
    data = image.reshape(-1)
    for layer in self.neural_net: data = layer.activate(data)
    return softmax(data)

  def train(self, iter: ImageIterator, learning_rate: Callable[['CNN', int], float], verbose: bool = True):
    while iter.has_next():
      image, label = iter.next()

