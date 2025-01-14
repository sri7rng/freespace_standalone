"""TensorRT engine utility functionality to make running inference with it easy.

This is a quasi copy of the tensorrt_engine part from the`onnx_tensorrt` python package. It exists here because the
actual `onnx_tensorrt` python package is currently only available in the TensorRT 6 Docker container and not easily
installable for the TensorRT 7 Docker container.

Link to Github source:
    https://github.com/onnx/onnx-tensorrt/blob/master/onnx_tensorrt/tensorrt_engine.py
"""
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Modified by Daimler AG and Robert Bosch GmbH
#
__copyright__ = """
COPYRIGHT: (c) 2018-2021 Daimler AG and Robert Bosch GmbH
The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages and can be prosecuted. All rights reserved
particularly in the event of the grant of a patent, utility model
or design.
"""


# pylint: skip-file
# flake8: noqa
# pydocstyle: noqa


import numpy as np
import pycuda.autoinit
import pycuda.driver
import pycuda.gpuarray
import tensorrt as trt
from six import string_types


class Binding(object):  # noqa
    def __init__(self, engine, idx_or_name):  # noqa
        if isinstance(idx_or_name, string_types):
            self.name = idx_or_name
            self.index = engine.get_binding_index(self.name)
            if self.index == -1:
                raise IndexError("Binding name not found: %s" % self.name)
        else:
            self.index = idx_or_name
            self.name = engine.get_binding_name(self.index)
            if self.name is None:
                raise IndexError("Binding index out of range: %i" % self.index)
        self.is_input = engine.binding_is_input(self.index)

        dtype = engine.get_binding_dtype(self.index)
        dtype_map = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF: np.float16,
            trt.DataType.INT8: np.int8,
            # trt.DataType.BOOL: np.bool  # TODO(blantho) Activate if only TensoRT 7 is used at some point
        }
        if hasattr(trt.DataType, "INT32"):
            dtype_map[trt.DataType.INT32] = np.int32

        self.dtype = dtype_map[dtype]
        shape = engine.get_binding_shape(self.index)

        self.shape = tuple(shape)
        # Must allocate a buffer of size 1 for empty inputs / outputs
        if 0 in self.shape:
            self.empty = True
            # Save original shape to reshape output binding when execution is done
            self.empty_shape = self.shape
            self.shape = tuple([1])
        else:
            self.empty = False
        self._host_buf = None
        self._device_buf = None

    @property
    def host_buffer(self):  # noqa
        if self._host_buf is None:
            self._host_buf = pycuda.driver.pagelocked_empty(self.shape, self.dtype)
        return self._host_buf

    @property
    def device_buffer(self):  # noqa
        if self._device_buf is None:
            self._device_buf = pycuda.gpuarray.empty(self.shape, self.dtype)
        return self._device_buf

    def get_async(self, stream):  # noqa
        src = self.device_buffer
        dst = self.host_buffer
        src.get_async(stream, dst)
        return dst


def squeeze_hw(x):  # noqa
    if x.shape[-2:] == (1, 1):
        x = x.reshape(x.shape[:-2])
    elif x.shape[-1] == 1:
        x = x.reshape(x.shape[:-1])
    return x


def check_input_validity(input_idx, input_array, input_binding):  # noqa
    # Check shape
    trt_shape = tuple(input_binding.shape)
    onnx_shape = tuple(input_array.shape)

    if onnx_shape != trt_shape:
        if not (trt_shape == (1,) and onnx_shape == ()):
            raise ValueError("Wrong shape for input %i. Expected %s, got %s." % (input_idx, trt_shape, onnx_shape))

    # Check dtype
    if input_array.dtype != input_binding.dtype:
        # TRT does not support INT64, need to convert to INT32
        if input_array.dtype == np.int64 and input_binding.dtype == np.int32:
            casted_input_array = np.array(input_array, copy=True, dtype=np.int32)
            if np.equal(input_array, casted_input_array).all():
                input_array = casted_input_array
            else:
                raise TypeError(
                    "Wrong dtype for input %i. Expected %s, got %s. Cannot safely cast."
                    % (input_idx, input_binding.dtype, input_array.dtype)
                )
        else:
            raise TypeError(
                "Wrong dtype for input %i. Expected %s, got %s." % (input_idx, input_binding.dtype, input_array.dtype)
            )
    return input_array


class Engine(object):  # noqa
    def __init__(self, trt_engine):  # noqa
        self.engine = trt_engine
        nbinding = self.engine.num_bindings

        bindings = [Binding(self.engine, i) for i in range(nbinding)]
        self.binding_addrs = [b.device_buffer.ptr for b in bindings]
        self.inputs = [b for b in bindings if b.is_input]
        self.outputs = [b for b in bindings if not b.is_input]

        for binding in self.inputs + self.outputs:
            _ = binding.device_buffer  # Force buffer allocation
        for binding in self.outputs:
            _ = binding.host_buffer  # Force buffer allocation
        self.context = self.engine.create_execution_context()
        self.stream = pycuda.driver.Stream()

    def __del__(self):  # noqa
        if self.engine is not None:
            del self.engine

    def run(self, inputs):  # noqa
        # len(inputs) > len(self.inputs) with Shape operator, input is never used
        # len(inputs) == len(self.inputs) for other operators
        if len(inputs) < len(self.inputs):
            raise ValueError("Not enough inputs. Expected %i, got %i." % (len(self.inputs), len(inputs)))
        if isinstance(inputs, dict):
            inputs = [inputs[b.name] for b in self.inputs]

        for i, (input_array, input_binding) in enumerate(zip(inputs, self.inputs)):
            input_array = check_input_validity(i, input_array, input_binding)
            input_binding_array = input_binding.device_buffer
            input_binding_array.set_async(input_array, self.stream)

        self.context.execute_async_v2(self.binding_addrs, self.stream.handle)

        results = [output.get_async(self.stream) for output in self.outputs]

        # For any empty bindings, update the result shape to the expected empty shape
        for i, (output_array, output_binding) in enumerate(zip(results, self.outputs)):
            if output_binding.empty:
                results[i] = np.empty(shape=output_binding.empty_shape, dtype=output_binding.dtype)

        self.stream.synchronize()
        return results

    def run_no_dma(self, batch_size):  # noqa
        self.context.execute_async(batch_size, self.binding_addrs, self.stream.handle)
