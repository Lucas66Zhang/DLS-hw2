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
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        #
        self.max = Z.max(axis=self.axes)
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        res = Z - max_Z
        res = array_api.exp(res)
        res = array_api.sum(res, axis=self.axes)
        res = array_api.log(res)
        res += max_Z.reshape(res.shape)
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs.realize_cached_data()
        Z -= array_api.max(Z, axis=self.axes, keepdims=True)
        Z = Tensor(Z)

        grad_1 = out_grad / summation(exp(Z), axis=self.axes)
        shape = [i for i in out_grad.shape]
        if self.axes:
            for axis in self.axes:
                shape.insert(axis, 1)
        else:
            shape = [1 for _ in range(len(out_grad.shape))]
        grad_2 = broadcast_to(reshape(grad_1, shape), Z.shape)
        return grad_2 * exp(Z)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

