from abc import ABCMeta, abstractmethod
from typing import Callable
import numpy as np
import pickle

from functions import ActivationFunction, Sigmoid, cross_entropy, cross_entropy_derivative, softmax, softmax_derivative
from image import ImageIterator, MNISTTrainIterator

class ConvolutionalLayer(metaclass=ABCMeta):
  @abstractmethod
  def convolve(self, image: np.ndarray) -> np.ndarray:
    raise NotImplementedError

  @abstractmethod
  # should randomly initialize the kernel
  def reset_kernel(self):
    raise NotImplementedError

  @abstractmethod
  # returns the kernel of this layer
  def get_kernel(self) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError

  @abstractmethod
  # set the kernel to the given value
  def set_kernel(self, kernel: np.ndarray):
    raise NotImplementedError

class NeuronLayer():
  def __init__(self, weights: np.ndarray, bias: np.ndarray, activation: ActivationFunction):
    self.weights = weights
    self.bias = bias
    self.activation_function = activation

  def reset_weights(self):
    self.weights = np.random.randn(self.weights.shape[0], self.weights.shape[1])

  def reset_bias(self):
    self.bias = np.zeros(self.bias.shape[0])

  def reset(self):
    self.reset_weights()
    self.reset_bias()

  def activate(self, data: np.ndarray) -> np.ndarray:
    return self.activation_function(np.dot(data, self.weights) + self.bias)

  # 
  def update_weights(self, input: np.ndarray, derivative: np.ndarray, output: np.ndarray, learning_rate: float) -> np.ndarray:
    """
    Updates the weights of this layer, returns the derivative for prior layer.
    """
    weights_delta = np.dot(output.T, derivative)
    self.weights -= learning_rate * weights_delta
    self.bias -= learning_rate * derivative
    return self.activation_function.derivative(input) * np.dot(derivative, self.weights)

class CNN():
  def __init__(self, convolution_layers: list[ConvolutionalLayer], neural_net: list[NeuronLayer]):
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

  def reset(self):
    for layer in self.convolution_layers: layer.reset_kernel()
    for layer in self.neural_net: layer.reset_weights()

  def train(self, iter: ImageIterator, learning_rate: Callable[['CNN', int], float], verbose: bool = True):
    n = 0
    while iter.has_next():
      image, label = iter.next()
      for cn in self.convolution_layers:
        image = cn.convolve(image)
      data = image.reshape(-1)

      deepnet_inputs: list[np.ndarray] = []
      for nl in self.neural_net:
        deepnet_inputs.append(data)
        data = nl.activate(data)

      softmax_input = data
      data = softmax(data)

      loss_input = data
      expected = np.zeros(10)
      expected[label] = 1
      data = cross_entropy(expected, data)

      if verbose: print(f"Loss: {np.sum(data):.4f}")
      lr = learning_rate(self, n)
      n += 1

      derivative = cross_entropy_derivative(expected, loss_input)
      derivative = softmax_derivative(softmax_input) * derivative
      output = softmax_input
      for nl in reversed(self.neural_net):
        input = deepnet_inputs.pop()
        derivative = nl.update_weights(input, derivative, output, lr)
        output = input


def _test_cnn():
  """
  Tests the CNN class.
  """
  cnn = CNN([], [
    NeuronLayer(np.zeros((28*28, 32)), np.zeros((28*28, 32)), Sigmoid()),
    NeuronLayer(np.zeros((32, 10)), np.zeros((32, 10)), Sigmoid()),
    NeuronLayer(np.zeros((10, 10)), np.zeros((10, 10)), Sigmoid()),
  ])

  cnn.reset()

  cnn.train(MNISTTrainIterator(), lambda cnn, i: 0.1, verbose=True)

if __name__ == '__main__':
  _test_cnn()

