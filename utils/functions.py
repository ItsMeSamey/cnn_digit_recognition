import numpy as np
import math

from numpy.matrixlib.defmatrix import mat

# functions from https://medium.com/machine-learning-researcher/artificial-neural-network-ann-4481fa33d85a

def cross_entropy(expected: np.ndarray, actual: np.ndarray) -> float:
  actual = np.clip(actual, 1e-15, np._NoValue)
  return -np.sum(expected * np.log(actual)) - np.sum((1 - expected) * np.log(1 - actual))

def cross_entropy_derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
  epsilon = 1e-15
  actual = np.clip(actual, epsilon, 1 - epsilon)
  return -(expected / actual) + ((1 - expected) / (1 - actual))

def softmax(input: np.ndarray) -> np.ndarray:
  input = np.exp(input)
  return input / np.sum(input)

def softmax_derivative(input: np.ndarray) -> np.ndarray:
  input = np.exp(input)
  sum = np.sum(input)
  return input * (sum - input) / sum

def sigmoid(val: float) -> float:
  return 1 / (1 + math.exp(-val))

def sigmoid_derivative(val: float) -> float:
  sig = sigmoid(val)
  return sig * (1 - sig)

def relu(val: float) -> float:
  return max(0, val)

def relu_derivative(val: float) -> float:
  return 1 if val > 0 else 0

def tanh(val: float) -> float:
  return math.tanh(val)

def tanh_derivative(val: float) -> float:
  return 1 / math.pow(math.cosh(val), 2)


