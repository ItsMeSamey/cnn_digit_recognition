from abc import ABCMeta, abstractmethod
import numpy as np
import pickle

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
    data = softmax(data)
    return self._activate(self._convolve(image))

