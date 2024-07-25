"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # return array_api.power(a, self.scalar)
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(out_grad,
                        mul_scalar(node.inputs[0] ** (self.scalar - 1),
                                   self.scalar))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # return array_api.divide(a, b)
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # quotient = dividend / divisor
        dividend, divisor = node.inputs
        return divide(out_grad, divisor), \
            negate(multiply(out_grad,
                            divide(dividend,
                                   power_scalar(divisor, 2))))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # WARNING
        # Dividing a float by an interger may introduce dtype mismatch. Fellows
        # on the forums reports `float32 / int` yields `float64`, although I
        # did not encounter this issue.
        #
        # Type alignment is pivotal in that optimizers shall not assign weights
        # of different type than the original one.
        # return array_api.divide(a, self.scalar,
        #                        dtype=a.dtype)
        return array_api.divide(a, self.scalar, dtype=a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # quotient = dividend / divisor
        return divide_scalar(out_grad, self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # `Transpose` behaves differently from `numpy.transpose`
        # in terms of input and default axes permutated.
        order = list(range(len(a.shape)))
        if self.axes:
            order[self.axes[0]], order[self.axes[1]] = order[self.axes[1]], order[self.axes[0]]
        else:
            order = order[:-2] + [order[-1], order[-2]]
        # return array_api.transpose(a, axes=tuple(order))
        return a.transpose(tuple(order))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad,
                       node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)  # .compact() # Why calling `compact`?

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        # Tensors are not subscriptable in needle.
        # Call `reshape` alternatively to add axes.
        singleton = list(range(len(self.shape) - len(in_shape))) + \
                    [i for i in range(-1, -len(in_shape) - 1, -1) if in_shape[i] == 1]

        return reshape(summation(out_grad, axes=tuple(singleton)), in_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if axes is None:
            self.axes = None
        elif isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Currently, Needle does not support multiple-axis summation in 1 pass.
        # Iterative summation along a single axis is adopted to bypass the
        # restriction. Multi-axis reduction will come in due time.
        # return array_api.summation(a, self.axes)
        # if self.axes and len(self.axes) > 1:
        #     # Upon each summation, 1 dimension is reduced. Hence, the operation
        #     # must be performed in a descending order of axes, which implies
        #     # the axes must be positive and sorted.
        #     self.axes = sorted([axis if axis > 0 else len(self.axes) + axis for axis in self.axes])
        #     for axis in self.axes[::-1]:
        #         a = array_api.sum(a, axis)
        #     return a
        # else:
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        axes_shape = list(node.inputs[0].shape)
        # `axes` must be a tuple, even if it is a single integer.
        # Otherwise, `self.axes` is considered FALSE when assigned 0.
        if self.axes:
            # if self.axes is not None:
            for i in self.axes:
                axes_shape[i] = 1
        else:
            axes_shape = [1, ] * len(axes_shape)
        return broadcast_to(reshape(out_grad, tuple(axes_shape)),
                            node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # return array_api.matmul(a, b)
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        out_shape, lhs_shape, rhs_shape = out_grad.shape, lhs.shape, rhs.shape

        return matmul(out_grad, transpose(rhs)) if len(lhs_shape) == len(out_shape) \
            else summation(matmul(out_grad, transpose(rhs)), \
                           axes=tuple(range(len(out_shape) - len(lhs_shape)))), \
 \
            matmul(transpose(lhs), out_grad) if len(rhs_shape) == len(out_shape) \
                else summation(matmul(transpose(lhs), out_grad), \
                               axes=tuple(range(len(out_shape) - len(rhs_shape))))
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide(out_grad, node.inputs[0])
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(out_grad,
                        exp(node.inputs[0]))
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #######################################################################
        # The original solution is not numerically stable.
        #
        # grad = divide(relu(node.inputs[0]), node.inputs[0])
        # return (multiply(out_grad, grad),)
        #######################################################################
        # There seems to be no numerically stable solution
        # that solely calls needle operations. assistance of
        # `array_api` is a must.
        node_input = node.inputs[0]
        return multiply(out_grad,
                        Tensor(node_input.realize_cached_data() > 0,
                               device=node.device,
                               dtype=node.dtype,
                               required_grad=node.requires_grad))
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
