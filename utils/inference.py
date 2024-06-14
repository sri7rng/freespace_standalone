"""Functions for running inference of TensorRT models.

.. note::

    Import the `InferenceEngine` always from this file because the actual implementation is hidden and can change.

"""
__copyright__ = """
===================================================================================
 C O P Y R I G H T
-----------------------------------------------------------------------------------
 Copyright (c) 2018-2021 Daimler AG and Robert Bosch GmbH. All rights reserved.
 Copyright (c) 2021-2023 Robert Bosch GmbH. All rights reserved.
 Copyright (c) 2023-2024 Robert Bosch GmbH and Cariad SE. All rights reserved.
===================================================================================
"""

import logging
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tensorrt as trt
from numpy.typing import NDArray

#from conversion.tensorrt.io import load_cuda_engine
#from conversion.tensorrt.utils.profiler import Profiler
#from conversion.tensorrt.utils.thirdparty.onnx_tensorrt.tensorrt_engine import Engine as InferenceEngine
from .tensorrt_engine import Engine as InferenceEngine

try:
    import giePlugins  # noqa: F401
except ImportError:
    logging.getLogger(__file__).warning(
        "Couldn't import custom TensorRT plugins. Your code will still run as intended but might fail if your model "
        "relies on custom plugins like NMSIndices or NMSGather. If you require support for custom plugins you can "
        "build them from the monorepo with `catkin build dl_giepluginspythoninterface`."
    )


# def load_and_run_inference(
#     trt_model_path: str,
#     inputs: Union[Sequence[NDArray[np.float32]], Dict[str, NDArray[np.float32]]],
#     trt_logger_severity: trt.Logger.Severity = trt.Logger.WARNING,
# ) -> Tuple[Dict[str, NDArray[np.float32]], Dict[str, Any]]:
#     """Load a CUDA engine (TensorRT model) and run inference on the given input values.

#     Args:
#         trt_model_path: Path to the saved TensorRT model (usually ends with '.gie')
#         inputs: Input values on which the inference is performed
#         trt_logger_severity: Minimum severity of messages logged from TensorRT

#     Returns:
#         Dictionary with output values,
#         Loaded metadata from the TensorRT saved model
#     """
#     cuda_engine, metadata = load_cuda_engine(trt_model_path, trt_logger_severity)
#     outputs = run_inference(cuda_engine, inputs)

#     return outputs, metadata


def run_inference(
    engine: Union[trt.ICudaEngine, InferenceEngine],
    inputs: Union[Sequence[NDArray[np.float32]], Dict[str, NDArray[np.float32]]],
) -> Dict[str, NDArray[np.float32]]:
    """Run inference with the given CUDA engine (TensorRT model) on the given input values.

    .. note::

        Choose to pass an instance of `InferenceEngine` if you are running inference several times (e.g. in a loop)
        because otherwise it gets created here and after several calls the GPU runs out of memory!

    Args:
        engine: CUDA engine (TensorRT model) or inference engine used for inference
        inputs: Input values on which the inference is performed

    Returns:
        Dictionary with output values
    """
    inference_engine = engine if isinstance(engine, InferenceEngine) else InferenceEngine(engine)  # type: ignore

    output_values = inference_engine.run(inputs)  # type: ignore

    return trt_output_as_dict(output_values, inference_engine)


def trt_output_as_dict(
    output_values: List[NDArray[np.float32]], inference_engine: InferenceEngine
) -> Dict[str, NDArray[np.float32]]:
    """Create dictionary indexed by output names."""
    return {output.name: output_value for output, output_value in zip(inference_engine.outputs, output_values)}


# def run_inference_and_measure_runtime(
#     engine: Union[trt.ICudaEngine, InferenceEngine],
#     inputs: Union[Sequence[NDArray[np.float32]], Dict[str, NDArray[np.float32]]],
#     iterations: int = 1000,
# ) -> Tuple[Dict[str, NDArray[np.float32]], float, pd.DataFrame]:
#     """Run inference with the given CUDA engine (TensorRT model) on the given input values and measure runtime.

#     Args:
#         engine: CUDA engine (TensorRT model) or inference engine used for inference
#         inputs: Input values on which the inference is performed
#         iterations: Number of calls to the engine from which the runtime is averaged

#     Returns:
#         Dictionary with output values,
#         Average runtime of a call in milliseconds (ms),
#         Detailed timings report for every layer of the engine
#     """
#     inference_engine = engine if isinstance(engine, InferenceEngine) else InferenceEngine(engine)  # type: ignore

#     profiler = Profiler()  # type: ignore
#     inference_engine.context.profiler = profiler

#     for _ in range(iterations):
#         output = run_inference(inference_engine, inputs)

#     runtime, raw_measurements = profiler.calculate_results()

#     return output, runtime, raw_measurements
