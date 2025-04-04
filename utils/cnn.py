from abc import ABCMeta, abstractmethod
import numpy as np

class ConvolutionalLayer(metaclass=ABCMeta):
  @abstractmethod
  def convolve(self, image: np.ndarray) -> np.ndarray:
    raise NotImplementedError

class FullyConnectedLayer(metaclass=ABCMeta):
  @abstractmethod
  def activate(self, data: np.ndarray) -> np.ndarray:
    raise NotImplementedError

class CNN():
  def __init__(self, convolution_layers: list[ConvolutionalLayer], neural_net: list[FullyConnectedLayer]):
    self.convolution_layers = convolution_layers
    self.neural_net = neural_net

  def _convolve(self, image: np.ndarray) -> np.ndarray:
    for layer in self.convolution_layers: image = layer.convolve(image)
    return image

  def _activate(self, data: np.ndarray) -> np.ndarray:
    for layer in self.neural_net: data = layer.activate(data)
    return data

  def forward(self, image: np.ndarray) -> np.ndarray:
    return self._activate(self._convolve(image))

