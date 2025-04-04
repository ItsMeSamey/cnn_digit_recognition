from abc import ABCMeta, abstractmethod
import time
from typing import Callable
import numpy as np
import pickle

from functions import ActivationFunction, ReLU,  Softmax,  cross_entropy, cross_entropy_derivative
from image import ImageIterator, MNISTTrainIterator

class ConvolutionalLayer(metaclass=ABCMeta):
  @abstractmethod
  def convolve(self, image: np.ndarray) -> np.ndarray:
    raise NotImplementedError

  def reset_kernel(self):
    self.kernel = np.random.randn(self.kernel.shape[0], self.kernel.shape[1])

  def reset_bias(self):
    self.bias = np.zeros(self.bias.shape[0])

  def reset(self):
    self.reset_kernel()
    self.reset_bias()

class NeuronLayer():
  def __init__(self, weights: np.ndarray, bias: np.ndarray, activation: ActivationFunction):
    self.weights = weights
    self.bias = bias
    self.activation_function = activation

  def reset_weights(self):
    self.weights = np.random.randn(self.weights.shape[0], self.weights.shape[1])
    self.weights /= np.max(np.abs(self.weights))

  def reset_bias(self):
    self.bias = np.zeros(self.bias.shape[0])

  def reset(self):
    self.reset_weights()
    self.reset_bias()

  def activate(self, data: np.ndarray) -> np.ndarray:
    print(f"Output: {np.dot(data, self.weights) + self.bias}")
    print(f"Activation: {self.activation_function(np.dot(data, self.weights) + self.bias)}")
    return self.activation_function(np.dot(data, self.weights) + self.bias)

  def update_weights(self, input: np.ndarray, derivative: np.ndarray, learning_rate: float) -> np.ndarray:
    """
    Updates the weights of this layer, returns the derivative for prior layer.
    """
    weights_delta = np.outer(input, derivative)

    self.weights -= learning_rate * weights_delta
    self.bias -= learning_rate * derivative

    return self.activation_function.derivative(input) + np.dot(self.weights, derivative)

class CNN():
  def __init__(self, convolution_layers: list[ConvolutionalLayer], neural_net: list[NeuronLayer]):
    self.convolution_layers = convolution_layers
    self.neural_net = neural_net

  def store(self) -> bytes:
    return pickle.dumps(self)

  @classmethod
  def load(cls, data: bytes) -> 'CNN':
    return pickle.loads(data)

  def reset(self):
    for layer in self.convolution_layers: layer.reset()
    for layer in self.neural_net: layer.reset()

  def forward(self, image: np.ndarray) -> np.ndarray:
    for layer in self.convolution_layers: image = layer.convolve(image)
    data = image.reshape(-1)
    for layer in self.neural_net: data = layer.activate(data)
    return data

  def train(self, iter: ImageIterator, learning_rate: Callable[['CNN', int], float], verbose: bool = True):
    n = 0
    while iter.has_next():
      time.sleep(0.1)
      image, label = iter.next()
      image = np.array(image, dtype=np.float32) / 255
      for cn in self.convolution_layers:
        image = cn.convolve(image)
      data = image.reshape(-1)

      deepnet_inputs: list[np.ndarray] = []
      for nl in self.neural_net:
        deepnet_inputs.append(data)
        data = nl.activate(data)

      loss_input = data
      expected = np.zeros(10)
      expected[label] = 1
      data = cross_entropy(expected, data)

      if verbose: print(f"Loss: {np.sum(data):.4f}")
      lr = learning_rate(self, n)
      n += 1

      derivative = cross_entropy_derivative(expected, loss_input)
      for nl in reversed(self.neural_net):
        input = deepnet_inputs.pop()
        derivative = nl.update_weights(input, derivative, lr)


def _test_cnn():
  """
  Tests the CNN class.
  """
  cnn = CNN([], [
    NeuronLayer(np.zeros((28*28, 32)), np.zeros((32)), ReLU()),
    NeuronLayer(np.zeros((32, 16)), np.zeros((16)), ReLU()),
    # NeuronLayer(np.zeros((16, 16)), np.zeros((16)), Sigmoid()),
    NeuronLayer(np.zeros((16, 10)), np.zeros((10)), Softmax()),
  ])

  cnn.reset()

  cnn.train(MNISTTrainIterator(), lambda cnn, i: 0.001, verbose=True)

if __name__ == '__main__':
  _test_cnn()

