"""Function saving and loading TensorRT CUDA engines."""
__copyright__ = """
===================================================================================
 C O P Y R I G H T
-----------------------------------------------------------------------------------
 Copyright (c) 2018-2021 Daimler AG and Robert Bosch GmbH. All rights reserved.
 Copyright (c) 2021-2022 Robert Bosch GmbH. All rights reserved.
 Copyright (c) 2023 Robert Bosch GmbH and Cariad SE. All rights reserved.
===================================================================================
"""

import logging
from typing import Any, Dict, Optional, Tuple

import tensorrt as trt
from utils.gieTypes import CGieHeader

from utils.definitions import (
    GIE_ATTRIBUTES_KEY,
    GIE_CONVERSION_DATE_KEY,
    GIE_CUDA_VERSION_KEY,
    GIE_CUDNN_VERSION_KEY,
    GIE_GPU_NAME_KEY,
    GIE_NETWORK_NAME_KEY,
    GIE_PREFIX_OUTPUT_ATTRIBUTE,
    GIE_TENSORRT_VERSION_KEY,
)

LOGGER = logging.getLogger(__name__)


def save_cuda_engine(
    cuda_engine: trt.ICudaEngine, trt_model_path: str, metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save a CUDA engine (TensorRT model) to a file together with the GIE header.

    Args:
        cuda_engine: Model to be saved
        trt_model_path: Path where to save the CUDA engine (TensorRT model)
        metadata: Metadata to be saved with the model (into the GIE header)
    """
    header = CGieHeader()

    if metadata is not None:
        for key, value in metadata.items():
            if key == GIE_NETWORK_NAME_KEY:
                header.networkName = metadata[GIE_NETWORK_NAME_KEY]
            elif key == GIE_ATTRIBUTES_KEY:
                header.attributes = metadata[GIE_ATTRIBUTES_KEY]
            elif key.startswith(GIE_PREFIX_OUTPUT_ATTRIBUTE):
                attribute_key = key[len(GIE_PREFIX_OUTPUT_ATTRIBUTE) + 1 : :]
                header.attributes[attribute_key] = value
            else:
                LOGGER.warning(f"Skipping unsupported metadata key '{key}' for gie header serialization.")

    with open(trt_model_path, "wb") as outfile:
        header.writeToFile(outfile)
        outfile.write(cuda_engine.serialize())


def load_cuda_engine(
    trt_model_path: str, trt_logger_severity: trt.Logger.Severity = trt.Logger.ERROR#trt.Logger.WARNING
) -> Tuple[trt.ICudaEngine, Dict[str, Any]]:
    """Load a CUDA engine including its metadata.

    Args:
        trt_model_path: Path to the saved TensorRT model (usually ends with '.gie')
        trt_logger_severity: Minimum severity of messages logged from TensorRT

    Returns:
        Loaded CUDA engine (TensorRT model),
        Loaded metadata from the saved model
    """
    with open(trt_model_path, "rb") as serialized_trt_model, trt.Runtime(trt.Logger(trt_logger_severity)) as runtime:
        gie_header = CGieHeader()
        gie_header.readFromFile(serialized_trt_model)
        cuda_engine = runtime.deserialize_cuda_engine(serialized_trt_model.read())

    metadata = {
        GIE_NETWORK_NAME_KEY: gie_header.networkName,
        GIE_ATTRIBUTES_KEY: gie_header.attributes,
        GIE_CUDA_VERSION_KEY: gie_header.cudaVersion,
        GIE_CUDNN_VERSION_KEY: gie_header.cudnnVersion,
        GIE_TENSORRT_VERSION_KEY: gie_header.gieVersion,
        GIE_GPU_NAME_KEY: gie_header.gpuName,
        GIE_CONVERSION_DATE_KEY: gie_header.conversionDate,
    }

    return cuda_engine, metadata
