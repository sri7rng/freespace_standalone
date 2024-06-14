"""Common functions manipulating onnx model."""
__copyright__ = """
===================================================================================
 C O P Y R I G H T
-----------------------------------------------------------------------------------
 Copyright (c) 2023-2024 Robert Bosch GmbH and Cariad SE. All rights reserved.
===================================================================================
"""

from typing import Dict, Mapping, NewType, Sequence, Tuple, Union

import onnx

OnnxInterface = NewType("OnnxInterface", Mapping[str, Mapping[str, Sequence[Union[int, str]]]])


def get_input_shapes_from_onnx(onnx_model: onnx.ModelProto) -> Dict[str, Tuple[int, ...]]:
    """Parse input name and input shape from onnx model.

    Args:
        onnx_model: onnx model to parse

    Returns:
        {input_name: input_shape_tuple}
    """
    input_shapes = {}
    for inp in onnx_model.graph.input:
        shape = tuple(int(d.dim_value) for d in inp.type.tensor_type.shape.dim)
        input_shapes[inp.name] = shape
    return input_shapes


def get_output_shapes_from_onnx(onnx_model: onnx.ModelProto) -> Dict[str, Tuple[int, ...]]:
    """Parse output name and output shape from onnx model.

    Args:
        onnx_model: onnx model to parse

    Returns:
        {output_name: output_shape_tuple}
    """
    output_shapes = {}
    for out in onnx_model.graph.output:
        shape = tuple(int(d.dim_value) for d in out.type.tensor_type.shape.dim)
        output_shapes[out.name] = shape
    return output_shapes


def get_interface_from_onnx(onnx_model: onnx.ModelProto) -> OnnxInterface:
    """Parse in-/output name and in-/output shape from onnx graph.

    Args:
        onnx_graph: onnx model to parse

    Returns:
        Onnx interface, meaning in-/output names and shapes
    """
    return OnnxInterface(
        {"inputs": get_input_shapes_from_onnx(onnx_model), "outputs": get_output_shapes_from_onnx(onnx_model)}
    )
