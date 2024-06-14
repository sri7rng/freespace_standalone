"""Functions for conversion of ONNX models to CUDA engines (TensorRT models).

How to create a CUDA engine (TensorRT model)
============================================

In order to create a CUDA engine (TensorRT model) using the functions in this file one needs a saved ONNX model. This
ONNX model can then be converted using the functions in this file.

Currently we create only TensorRT models with fixed shapes of all inputs including batch size. The ONNX model itself can
have variable input shapes (if all TensorRT operations in that specific model support it), however fixed shapes are used
for the exported TensorRT model and its optimization.

Let's look at a minimal example how to perform the conversion:

.. code::

    onnx_model_path = "path/to/some/model.onnx"
    trt_model_path = "path/to/some/model.gie"

    # Input shapes must be set for each input of the model, keys are input names from the ONNX/TensorFlow saved model
    input_shapes = {
        "input_1": (1, 3, 480, 640),
        "input_2": (1, 10, 20),
    }

    # Run the conversion and save the CUDA engine (TensorRT model)
    convert_onnx_model_to_tensorrt(onnx_model_path, input_shapes, trt_model_path)


Numeric precision and calibration
---------------------------------

In order to speed up the inference, the CUDA engine execution (TensorRT model) can be configured to use other types of
CUDA cores (like int8). To do that, one needs to specify the numeric precision (and if needed a calibrator) for the
conversion procedure.

The conversion function currently supports three different precisions: float32, float16, int8. Float32 is the default,
float16 can be configured via a parameter, and int8 is activated by passing an `Int8Calibrator` to the conversion
function:

.. code::

    # Activate float16 and int8 calibration by setting a flag and passing a calibrator
    convert_onnx_model_to_tensorrt(..., use_fp16_precision=True, int8_calibrator=MyInt8Calibrator())

"""
from __future__ import annotations

__copyright__ = """
===================================================================================
 C O P Y R I G H T
-----------------------------------------------------------------------------------
 Copyright (c) 2018-2021 Daimler AG and Robert Bosch GmbH. All rights reserved.
 Copyright (c) 2021-2023 Robert Bosch GmbH. All rights reserved.
 Copyright (c) 2023-2024 Robert Bosch GmbH and Cariad SE. All rights reserved.
===================================================================================
"""

import fcntl
import json
import logging
import os
from typing import Any, Sequence

import tensorrt as trt

from .Int8_calib import Int8Calibrator
from cuda_handler import save_cuda_engine

_logger = logging.getLogger(__file__)

try:
    import giePlugins  # noqa: F401   # pylint: disable=W0611

    CUSTOM_PLUGINS_AVAILABLE = True
except ImportError:
    CUSTOM_PLUGINS_AVAILABLE = False
    _logger.warning(
        "Couldn't import custom TensorRT plugins. Your code will still run as intended but might fail if your model "
        "relies on custom plugins like NMSIndices or NMSGather. If you require support for custom plugins you can "
        "build them from the monorepo with `catkin build dl_giepluginspythoninterface`."
    )


def convert_onnx_model_to_tensorrt(
    onnx_model_path: str,
    input_shapes: dict[str, Sequence[int]],
    trt_model_path: str | None = None,
    max_workspace_size: int = 2 * 2**30,
    metadata: dict[str, Any] | None = None,
    int8_calibrator: Int8_calib | None = None,
    use_fp16_precision: bool = False,
    force_int8_inputs: bool = False,
    force_int8_outputs: bool = False,
    trt_logger_severity: trt.Logger.Severity = trt.Logger.WARNING,
    algorithm_selector: trt.IAlgorithmSelector | None = None,
    timing_cache_path: str | None = None,
) -> trt.ICudaEngine:
    """Converts an ONNX model into a TensorRT inference model (and optionally saves it to a file).

    Args:
        onnx_model_path: Path to the ONNX model to be converted
        input_shapes: Mapping from input names to fixed shapes (shapes including batch size)
        trt_model_path: Path where the TensorRT model will be saved (usually with  `.gie` extension)
        max_workspace_size: Maximum TensorRT converter workspace size (in B, default 1GB)
        metadata: Metadata to be saved with the model (e.g. network name or attributes)
        int8_calibrator: Calibrator used for INT8 precision, no calibration is done if not supplied
        use_fp16_precision: If FP16 precision operations should be allowed
        force_int8_inputs: Make all inputs INT8
        force_int8_outputs: Make all outputs INT8
        trt_logger_severity: Minimum severity of messages logged from TensorRT
        algorithm_selector: algorithm selector for TensorRT engine builder.
        timing_cache_path: Path to the TensorRT timing cache.

    Returns:
        The converted CUDA engine (TensorRT model)
    """
    trt_logger = trt.Logger(trt_logger_severity)

    # The onnx parser version we currently use requires this to be used
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(trt_logger) as builder, builder.create_network(explicit_batch) as network:
        with trt.OnnxParser(network, trt_logger) as onnx_parser, builder.create_builder_config() as config:
            if timing_cache_path is not None:
                if os.path.exists(timing_cache_path):
                    with open(timing_cache_path, "rb") as f:
                        fcntl.lockf(f, fcntl.LOCK_SH)
                        cache = config.create_timing_cache(f.read())
                        fcntl.lockf(f, fcntl.LOCK_UN)
                    config.set_timing_cache(cache, ignore_mismatch=False)
                else:
                    cache = config.create_timing_cache(b"")
                    config.set_timing_cache(cache, ignore_mismatch=False)

            _parse_onnx_model(onnx_model_path, onnx_parser)

            # Since we don't currently use dynamic shapes, the input shapes of the network must be fixed in order to
            # get a valid TensorRT engine that can be used for inference
            _fix_input_shapes(network, input_shapes)

            config.max_workspace_size = max_workspace_size
            config.add_optimization_profile(_create_optimization_profile(builder, network, input_shapes))
            builder.max_batch_size = max(input_shape[0] for input_shape in input_shapes.values())

            if int8_calibrator is not None:
                _logger.info(f"INT8 calibration enabled for input shapes {input_shapes}")
                int8_calibrator.initialize(_order_input_shapes(network, input_shapes))
                config.int8_calibrator = int8_calibrator
                config.flags = config.flags | 1 << (int)(trt.BuilderFlag.INT8)

            if use_fp16_precision:
                config.flags = config.flags | 1 << (int)(trt.BuilderFlag.FP16)

            if force_int8_inputs:
                for i in range(network.num_inputs):
                    input = network.get_input(i)
                    input.dtype = trt.DataType.INT8

            if force_int8_outputs:
                for i in range(network.num_outputs):
                    output = network.get_output(i)
                    output.dtype = trt.DataType.INT8

            config.algorithm_selector = algorithm_selector
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

            cuda_engine = builder.build_engine(network, config)

            if timing_cache_path is not None:
                cache = config.get_timing_cache()
                with cache.serialize() as buffer:
                    with open(timing_cache_path, "wb") as f:
                        fcntl.lockf(f, fcntl.LOCK_EX)
                        f.write(buffer)
                        f.flush()
                        os.fsync(f)
                        fcntl.lockf(f, fcntl.LOCK_UN)

    if cuda_engine is None:
        raise RuntimeError(
            f"Failed to build TensorRT engine from '{onnx_model_path}' ONNX model! Unfortunately I cannot tell you "
            "more, you have to look at the TensorRT error log. You might also want to increase TensorRT log level "
            "in case of unclear logs."
        )

    if trt_model_path is not None:
        save_cuda_engine(cuda_engine, trt_model_path, metadata)

        # Save engines layer information in human readable json next to the engine file
        inspector = cuda_engine.create_engine_inspector()
        engine_layer_info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))

        engine_layer_info_file = f"{os.path.splitext(trt_model_path)[0]}_layer_info.json"
        with open(engine_layer_info_file, "w", encoding="utf-8") as file_handle:
            json.dump(engine_layer_info, file_handle, indent=4)

    return cuda_engine


def _parse_onnx_model(onnx_model_path: str, onnx_parser: trt.OnnxParser) -> None:
    """Parse the given ONNX model into the given TensorRT network.

    Args:
        onnx_model_path: Path to the ONNX model to be converted
        onnx_parser: Instance of an ONNX parser created with a TensorRT builder network
    """
    with open(onnx_model_path, "rb") as infile:
        if not onnx_parser.parse(infile.read()):

            errors = ""
            for i in range(onnx_parser.num_errors):
                errors += f"    [{i}] {onnx_parser.get_error(i)}\n"

            raise OSError(f"Errors found when trying to parse the '{onnx_model_path}' ONNX model:\n" + errors)


def _fix_input_shapes(trt_network: trt.INetworkDefinition, input_shapes: dict[str, Sequence[int]]) -> None:
    """Sets all input shapes of the network to the provided values.

    .. note::

        This is needed because if the TensorRT model is supposed to be saved with explicit input shapes they have to be
        set in the TensorRT network (otherwise the inference will not work).

    Args:
        trt_network: Network for which the shapes are going to be rewritten
        input_shapes: Dictionary of tuples specifying the input shapes for each input (including batch size)
    """
    for i in range(trt_network.num_inputs):
        trt_input = trt_network.get_input(i)

        if trt_input.name not in input_shapes:
            raise KeyError(f"No input shape is set for '{trt_input.name}' input!")

        input_shape = input_shapes[trt_input.name]

        if len(input_shape) != len(trt_input.shape):
            raise ValueError(
                f"The configured input shape '{input_shape}' for input '{trt_input.name}' does not have the same number"
                f" of dimensions as in the loaded ONNX model '{trt_input.shape}'!"
            )

        for j, input_shape_j in enumerate(input_shape):
            if trt_input.shape[j] not in {-1, None} and input_shape_j != trt_input.shape[j]:
                raise ValueError(
                    f"The dimension '{j}' of the input '{trt_input.name}' '{trt_input.shape}' is not dynamic, but is "
                    f"overwritten in the TensorRT conversion configuration to '{input_shape_j}' '{input_shape}'!"
                )

        trt_input.shape = trt.Dims(input_shape)


def _create_optimization_profile(
    trt_builder: trt.Builder, trt_network: trt.INetworkDefinition, input_shapes: dict[str, Sequence[int]]
) -> trt.IOptimizationProfile:
    """Creates an optimization profile used for TensorRT optimization of a model.

    .. note::

        Always creates a profile with fixed input shape (min == opt == max).

    Args:
        trt_builder: A TensorRT builder used for the conversion
        trt_network: A TensorRT network created by that builder
        input_shapes: Dictionary of tuples specifying the input shapes for each input

    Returns:
        Optimization profile with all input shapes set to the given values
    """
    profile = trt_builder.create_optimization_profile()

    for i in range(trt_network.num_inputs):
        trt_input = trt_network.get_input(i)

        input_shape = input_shapes[trt_input.name]

        profile.set_shape(trt_input.name, min=input_shape, opt=input_shape, max=input_shape)

    return profile


def _order_input_shapes(
    trt_network: trt.INetworkDefinition, input_shapes: dict[str, Sequence[int]]
) -> dict[str, Sequence[int]]:
    """Order input shapes dictionary in the order expected by the TensorRT model.

    .. note::
        This is needed in order to define the data buffers in the int8 calibration process in the correct order.

    Args:
        trt_network: Network for which the shapes are going to be rewritten
        input_shapes: Dictionary of tuples specifying the input shapes for each input (including batch size)

    Returns:
        ordered_shapes: Dictionary of tuples specifying the input shapes for each input in the order expected by the
            TensorRT model
    """
    ordered_shapes = dict()
    for i in range(trt_network.num_inputs):
        trt_input = trt_network.get_input(i)

        if trt_input.name not in input_shapes:
            raise KeyError(f"No input shape is set for '{trt_input.name}' input!")

        ordered_shapes[trt_input.name] = input_shapes[trt_input.name]

    return ordered_shapes
