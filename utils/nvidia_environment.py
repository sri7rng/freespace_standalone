"""This module contains helpers to gather information about the hard- and software runtime of the calling process."""

__copyright__ = """
===================================================================================
 C O P Y R I G H T
-----------------------------------------------------------------------------------
 Copyright (c) 2023 Robert Bosch GmbH and Cariad SE. All rights reserved.
===================================================================================
"""

import csv
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass
class NvidiaGpuInfo:
    """Contains information about Nvidia based GPUs."""

    name: str
    bus_id: str
    memory_total: int
    vbios_version: str
    driver_version: str
    max_pci_link_gen: str

    @staticmethod
    def gather() -> List["NvidiaGpuInfo"]:
        """Gather needed information to build a list of `NvidiaGpuInfo` objects.

        Returns:
            A list of `NvidiaGpuInfo` objects representing the available Nvidia GPUs on this machine.
        """

        def _to_kwargs(csv_row: Dict[str, Any]) -> Dict[str, Any]:
            kwargs: Dict[str, Any] = {k.strip(): v.strip() for k, v in csv_row.items()}

            name_mapping = {
                "pci.bus_id": "bus_id",
                "memory.total [MiB]": "memory_total",
                "pcie.link.gen.max": "max_pci_link_gen",
            }
            for csv_name, arg_name in name_mapping.items():
                kwargs[arg_name] = kwargs[csv_name]
                del kwargs[csv_name]

            # MiB to Bytes
            kwargs["memory_total"] = int(kwargs["memory_total"].split(" ")[0]) * 1024**2

            return kwargs

        # we bail out if nvidia-smi does not exist
        if shutil.which("nvidia-smi") is None:
            return []

        cmd_args = [
            "nvidia-smi",
            "--query-gpu=gpu_name,gpu_bus_id,vbios_version,memory.total,driver_version,pcie.link.gen.max",
            "--format=csv",
        ]
        process = subprocess.run(args=cmd_args, text=True, capture_output=True, check=True)

        csv_output = StringIO(process.stdout)
        reader = csv.DictReader(csv_output, delimiter=",")
        gpus = [NvidiaGpuInfo(**_to_kwargs(row)) for row in reader]

        return gpus


@dataclass
class PythonRuntimeInfo:
    """Contains information about the current python runtime environment."""

    interpreter_path: Path
    version: str
    packages: List[str]

    @staticmethod
    def gather() -> "PythonRuntimeInfo":
        """Gather needed information to build a `PythonRuntimeInfo` object.

        Returns:
            A `PythonRuntimeInfo` object representing information about the python environment which runs this
            interpreter.
        """

        def _gather_packages() -> List[str]:
            cmd_args = ["pip", "freeze"]
            process = subprocess.run(args=cmd_args, text=True, capture_output=True, check=True)
            return process.stdout.splitlines(keepends=False)

        version_info = sys.version_info
        return PythonRuntimeInfo(
            interpreter_path=Path(sys.executable),
            version=f"{version_info.major}.{version_info.minor}.{version_info.micro}-{version_info.releaselevel}",
            packages=sorted(_gather_packages()),
        )


@dataclass
class MemoryInfo:
    """Contains information about the amount of memory on the system."""

    mem_total: int
    mem_free: int
    mem_available: int
    swap_total: int
    swap_free: int

    @staticmethod
    def gather() -> "MemoryInfo":
        """Gather needed information to build a `MemoryInfo` object.

        Returns:
            A `MemoryInfo` object representing information about the system memory (RAM).
        """

        if platform.system() != "Linux":
            raise NotImplementedError("MemoryInfo currently only supports Linux!")

        meminfo_file = Path("/proc/meminfo")

        # reads the meminfo file line by line and converts the lines like 'MemTotal:       65646144 kB' into a dict
        # mapping the name to the value
        str_pairs = {
            t[0]: t[1]
            for t in [
                [e.strip() for e in line.split(":", maxsplit=1)] for line in meminfo_file.read_text().splitlines()
            ]
        }

        meminfo = {k: int(v.replace(" kB", "")) * 1024 for k, v in str_pairs.items()}

        return MemoryInfo(
            meminfo["MemTotal"],
            meminfo["MemFree"],
            meminfo["MemAvailable"],
            meminfo["SwapTotal"],
            meminfo["SwapFree"],
        )


@dataclass
class CpuInfo:
    """Contains information about the CPUs used by the system and the current load average."""

    cpu_count: int
    load1: float
    load5: float
    load15: float

    @staticmethod
    def gather() -> "CpuInfo":
        """Gather needed information to build a `CpuInfo` object.

        Returns:
            A `CpuInfo` object representing information about the CPU used in this machine.
        """

        if platform.system() != "Linux":
            raise NotImplementedError("CpuInfo currently only supports Linux!")

        cpu_count = os.cpu_count()
        if not cpu_count:
            cpu_count = 0

        load1, load5, load15 = os.getloadavg()

        return CpuInfo(cpu_count, load1, load5, load15)


@dataclass
class MountInfo:
    """Contains information about the mounting locations on the local system."""

    path: Path
    total: int
    used: int
    free: int

    @classmethod
    def gather(cls) -> List["MountInfo"]:
        """Gather needed information to build a list of `MountInfo` objects.

        Returns:
            A list of `MountInfo` objects representing information about mounted disks on the local machine.
        """

        def startswith_one_of(value: str, starts: Sequence[str]) -> bool:
            for s in starts:
                if value.startswith(s):
                    return True
            return False

        if platform.system() != "Linux":
            raise NotImplementedError("MountInfo currently only supports Linux!")

        mounts_file = Path("/proc/mounts")
        mounts = [line.split(" ") for line in mounts_file.read_text().splitlines()]

        ignored_roots = ["/proc", "/sys", "/dev"]
        mount_paths = {Path(m[1]) for m in mounts if not startswith_one_of(m[1], ignored_roots)}

        mount_infos = []
        for path in mount_paths:
            try:
                total, used, free = shutil.disk_usage(path)
                mount_infos.append(MountInfo(path, total, used, free))
            except PermissionError:
                pass
            except OSError:
                pass

        return mount_infos


@dataclass
class EnvironmentInfo:
    """Contains all available information about the soft- and hardware environment."""

    runtime: PythonRuntimeInfo
    gpus: List[NvidiaGpuInfo]
    cpus: CpuInfo
    memory: MemoryInfo
    mounts: List[MountInfo]

    @classmethod
    def gather(cls) -> "EnvironmentInfo":
        """Gather needed information to build a `EnvironmentInfo` object.

        Currently only Linux is fully supported for this and if this function gets called from another OS a
        `NotImplementedError` gets raised!

        Returns:
            A `EnvironmentInfo` object representing the hard and software environment which executes this function call.
        """
        return EnvironmentInfo(
            runtime=PythonRuntimeInfo.gather(),
            gpus=NvidiaGpuInfo.gather(),
            cpus=CpuInfo.gather(),
            memory=MemoryInfo.gather(),
            mounts=MountInfo.gather(),
        )
