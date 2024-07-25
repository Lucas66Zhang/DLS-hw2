from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if axes is None:
            self.axes = None
        elif isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        #
        self.max = Z.max(axis=self.axes)
        # `self.axes` must be a tuple, even if it is a single integer.
        # Otherwise, `self.axes` is considered FALSE when assigned 0.
        # if self.axes:
        if self.axes is not None:
            # broadcast `z_max` to shape of `Z`
            reduced_shape = list(Z.shape)
            for i in self.axes:
                reduced_shape[i] = 1
            self.reduced_shape = tuple(reduced_shape)
        else:
            self.reduced_shape = (1,) * len(Z.shape)
        return self.max + \
            array_api.log(
                array_api.sum(
                    array_api.exp(
                        Z - array_api.broadcast_to(
                            array_api.reshape(
                                self.max, self.reduced_shape
                            ), Z.shape
                        )
                    ), self.axes
                )
            )
        # else:
        #   self.reduced_shape = (1,) * len(Z.shape)
        #   return self.max + array_api.log(array_api.summation(array_api.exp(Z - self.max)))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        exp_z = exp(node.inputs[0] - reshape(Tensor(self.max,
                                                    dtype=node.inputs[0].dtype,
                                                    requires_grad=False),
                                             self.reduced_shape))
        # Needle summation does not preserve original shape.
        sum_z = reshape(summation(exp_z, self.axes),
                        self.reduced_shape)
        out_grad = reshape(out_grad, self.reduced_shape)

        return multiply(out_grad,
                        divide(exp_z, sum_z))
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

