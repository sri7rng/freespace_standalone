"""Module that contains the int8 calibrators for tensorrt conversion."""
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

import logging
import os
import sys
from typing import Any, Generator, Sequence

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """Base class for int8 calibration."""

    def __init__(self, cache_path: str):
        """Initialize the int8 calibrator.

        Args:
            cache_path: File to read/write calibration cache from/to
        """
        super().__init__()

        self.device_memory: dict[str, Any] | None = None
        self.input_shapes: dict[str, Sequence[int]] | None = None
        self.batch_size: int | None = None
        self.sample_generator: Generator[dict[str, NDArray[Any]], None, None] | None = None
        self.cache_path = cache_path

    def _generate_input_sample(self) -> Generator[dict[str, NDArray[Any]], None, None]:
        """Generate input samples for each network input.

        Note:
            It is crucial that the returned data fits the network input shapes.

        Yields:
            One input sample per network input (without batch dimension)
        """
        raise NotImplementedError("Calibrator data generation has not been implemented!")

    def initialize(self, input_shapes: dict[str, Sequence[int]]) -> None:
        """Initialize the int8 calibrator with the network input shapes.

        Args:
            input_shapes: Mapping from input names to input shapes of the network (including batch dimension)
        """
        self.input_shapes = input_shapes

        # Infer batch size from input shape
        any_input_name = next(iter(input_shapes))
        self.batch_size = input_shapes[any_input_name][0]  # Assuming batch size is first dimension

        # Allocate device memory for each input
        self.device_memory = {
            name: cuda.mem_alloc(np.empty(shape, dtype=np.float32).nbytes) for name, shape in self.input_shapes.items()
        }

        # Create the sample generator - because the `get_batch()` method is not a generator on its own
        self.sample_generator = self._generate_input_sample()

    def get_batch(self, names: list[str]) -> list[int]:
        """Load a batch for each network input into device memory and return the device memory pointers.

        Args:
            names: List of input names of the network

        Returns:
            List of device pointers in the same order as the given input names
        """
        if not (
            self.device_memory is not None
            and self.input_shapes is not None
            and self.batch_size is not None
            and self.sample_generator is not None
        ):
            raise (ValueError("The calibrator has to be initialized! Try calling initialize() before using it."))
        assert set(names) == set(
            self.input_shapes.keys()
        ), f"Initialized inputs {set(self.input_shapes.keys())} have different names than requested {set(names)}!"

        # Try to gather all input data
        samples = []
        for _ in range(self.batch_size):
            try:
                sample = next(self.sample_generator)
                self._check_sample(sample)
            except StopIteration:
                return []  # Return empty list to signal that all data has been processed
            except (KeyError, AssertionError):
                raise  # Re-raise exceptions known NOT to be swallowed by TensorRT
            except Exception as exception:
                # Some exceptions get swallowed by TensorRT, leading to the same stupid error message 'get_batch() takes
                # 2 positional arguments but 3 were given'. Therefore, the errors are logged here and program exited to
                # avoid this message and give the actual error message
                LOGGER.exception(exception)
                sys.exit(1)

            samples.append(sample)

        # Build a separate batch for each network input and copy it to device memory
        device_pointers = []
        for name in names:
            batch = np.ascontiguousarray([sample[name] for sample in samples])
            cuda.memcpy_htod(self.device_memory[name], batch)

            device_pointers.append(int(self.device_memory[name]))

        return device_pointers

    def get_batch_size(self) -> int:
        """Get the batch size used for calibration batches.

        DO NOT USE THIS METHOD FOR ANYTHING!

        Note:
            The value returned by this function should not be used anywhere in your code
            as it does not reflect the batch size of the network, or even the batch size
            used by this calibrator implementation!

            We have some speculations why this value needs to be fixed to 1:
            In the IExecutionContext API there is an execute and an executeV2 function. According to the
            docu the latter should presumably be used for explicit batch networks, but it isn't.
            Instead execute is called, which (in contrast to executeV2) takes the batch size as an argument
            and presumably sets the expected shapes accordingly.
            By returning the same batch size in here we would double down on this value and the calibrator
            would expect an input that is bigger than what we want by a factor that equals our the batch size.
            This problem is also pointed out by the following tensorrt warning during calibration:
            'Explicit batch network detected and batch size specified, use execute without batch size instead.'
            We have tests in place that should break if this is ever fixed in tensorrt.
        """
        return 1

    def read_calibration_cache(self) -> bytes | None:
        """Return an existing calibration cache if available, instead of calibrating again."""
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                LOGGER.info(f"Using existing calibration cache to save time: {self.cache_path}")
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Write the calibration cache to the cache file location.

        Args:
            cache: Data pointer to cache
        """
        with open(self.cache_path, "wb") as f:
            LOGGER.info(f"Creating calibration cache for future use: {self.cache_path}")
            f.write(cache)

    def _check_sample(self, sample: dict[str, NDArray[Any]]) -> None:
        """Check that the given data sample provides correct data for all the network inputs.

        Args:
            sample: Dict of network input names mapping to the respective input data
        """
        assert self.input_shapes is not None, "The calibrator has to be initialized!"
        for input_name, input_shape in self.input_shapes.items():
            if input_name not in sample:
                raise KeyError(
                    f"Generated sample has no data for input '{input_name}', only data for '{list(sample.keys())}' "
                    f"inputs provided!"
                )

            if sample[input_name].dtype != np.float32:
                raise ValueError(
                    f"Data for input '{input_name}' has dtype '{sample[input_name].dtype}', but float32 is required!"
                )

            if sample[input_name].shape != input_shape[1:]:
                raise ValueError(
                    f"Data for input '{input_name}' has shape '{sample[input_name].shape}', but should be "
                    f"'{input_shape[1:]}'!"
                )
