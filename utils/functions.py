from abc import ABCMeta, abstractmethod
import numpy as np

# functions from https://medium.com/machine-learning-researcher/artificial-neural-network-ann-4481fa33d85a

def cross_entropy(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
  actual = np.clip(actual, 1e-5, None)
  return -expected * np.log(actual) - (1 - expected) * np.log(1 - actual)

def cross_entropy_derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
  epsilon = 1e-5
  actual = np.clip(actual, epsilon, 1 - epsilon)
  return -(expected / actual) + ((1 - expected) / (1 - actual))

class ActivationFunction(metaclass=ABCMeta):
  @abstractmethod
  def __call__(self, input: np.ndarray) -> np.ndarray:
    raise NotImplementedError

  @abstractmethod
  def derivative(self, input: np.ndarray) -> np.ndarray:
    raise NotImplementedError

class Softmax(ActivationFunction):
  def __call__(self, input: np.ndarray) -> np.ndarray:
    input = np.exp(input)
    return input / np.sum(input)

  def derivative(self, input: np.ndarray) -> np.ndarray:
    input = np.exp(input)
    sum = np.sum(input)
    return np.full(input.shape, (1/sum) * (1 - 1/sum))

class Sigmoid(ActivationFunction):
  def __call__(self, input: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-input))

  def derivative(self, input: np.ndarray) -> np.ndarray:
    sig = self(input)
    return sig * (1 - sig)

class ReLU(ActivationFunction):
  def __call__(self, input: np.ndarray) -> np.ndarray:
    return np.maximum(0, input)

  def derivative(self, input: np.ndarray) -> np.ndarray:
    return np.full(input.shape, (input > 0).astype(np.float32))

class Tanh(ActivationFunction):
  def __call__(self, input: np.ndarray) -> np.ndarray:
    return np.tanh(input)

  def derivative(self, input: np.ndarray) -> np.ndarray:
    return 1 / np.power(np.cosh(input), 2)

