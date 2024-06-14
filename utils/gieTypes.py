#!/usr/bin/python
from __future__ import print_function, absolute_import

__copyright__ = """
COPYRIGHT: (c) 2017-2021 Daimler AG and Robert Bosch GmbH
The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages and can be prosecuted. All rights reserved
particularly in the event of the grant of a patent, utility model
or design.
"""

import os
import tempfile
import unittest
import logging

_logger = logging.getLogger(__name__)


class CGieHeader(object):
    """
    CGieHeader matching the C++ struct SGieStreamHeader
    """

    def __init__(self, modelFile=None, raiseOnMissmatch=True):
        super(CGieHeader, self).__init__()
        # version of the header, uint32_t
        self.version = 5

        # cuda version
        self.cudaVersion = 0

        # cudnn version
        self.cudnnVersion = 0

        # version of TensorRt/Gie, uint32_t
        self.gieVersion = 0

        # get name of the GPU
        self.gpuName = 0

        # the name of the network, string
        self.networkName = "unknown"

        # conversion date
        self.conversionDate = 0

        # the output attributes, a.k.a. channel names
        # keys are the output buffer names, values the attributes
        # std::map<std::string,std::string>
        self.attributes = {}

        # argmax image layers
        # keys are the input buffer names to these layers,
        # values are the output buffer names
        # std::map<std::string,std::string>
        self.argmaxImageLayers = {}

        # mean values
        # keys are input buffer names of the network
        # values are the corresponding mean vectors
        # the ordering of the mean values is always with respect
        # to the network. That means, if the input buffer
        # is a 3 channel buffer with BGR channel order, then
        # the mean values are also in the BGR order.
        # std::map< std::string,std::vector<float> >
        self.meanValues = {}

        # if a model file is provided, read data from there
        if modelFile:
            with open(modelFile, "rb") as f:
                # if missmatches in config need to be detected, first determine system config
                if raiseOnMissmatch:
                    self._initFromSystem()
                # now, read from file
                self.readFromFile(f, raiseOnMissmatch)
        else:
            # if no model file was provided, initialize values from system config
            self._initFromSystem()

    def _initFromSystem(self):
        """
        Initialize everything from system configuration

        Sets all values from system library and hardware values
        """
        import re
        import glob
        import time
        import tensorrt as trt
        import subprocess
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa F401

        # cuda version (single integer as CUDART_VERSION defined in cuda_runtime_api.h)
        cudaVersion = cuda.get_version()
        cudaVersion = cudaVersion[0] * 1000 + cudaVersion[1] * 10 + cudaVersion[2]
        self.cudaVersion = cudaVersion

        # cudnn version
        # TODO (erehder): this is extremely hacky, please feel free to find a nicer way
        # find TRT module folder
        trt_folder = os.path.dirname(trt.__file__)
        # find all so files
        try:
            so_files = glob.iglob("{}/**/*.so".format(trt_folder), recursive=True)
        except TypeError:
            so_files = glob.glob("{}/*.so".format(trt_folder)) + glob.glob("{}/**/*.so".format(trt_folder))
        # set empty path to cudnn libraray
        cudnn_path = None
        # pattern to match cudnn
        parser = re.compile(r"^libcudnn.* => (.*) \(")
        # go through all libraries in tensorrt module
        for library in so_files:
            # look at dynamic library links
            links = subprocess.check_output(["ldd", library]).decode("utf-8")
            # go through all libs
            for linked_lib in links.split("\n"):
                # see if link is cudnn
                linked_lib = linked_lib.strip()
                matches = parser.match(linked_lib)
                if matches:
                    # if matched, extract the file link
                    cudnn_path = matches.groups()[0]
                    break
            if cudnn_path:
                break
        else:
            raise IOError("Can't find link to cudnn in tensorrt libraries")
        # follow the link to the actual file
        cudnn_path = os.path.realpath(cudnn_path)
        cudnn_re = re.findall(r"/opt/athena/cudnn/(\d+)\.(\d+)\.(\d+)/", cudnn_path)

        if cudnn_re:
            cudnn_version = [int(v) for v in cudnn_re[0]]
        else:
            # try to split. if this fails, well....
            cudnn_version = [int(v) for v in cudnn_path.rsplit(".", 3)[1:]]
        # stitch version string
        self.cudnnVersion = cudnn_version[0] * 1000 + cudnn_version[1] * 100 + cudnn_version[2]

        # version of TensorRt/Gie, uint32_t
        gieVersion = [int(v) for v in trt.__version__.split(".")]
        gieVersion = gieVersion[0] * 1000 + gieVersion[1] * 100 + gieVersion[2]
        self.gieVersion = gieVersion

        # get name of the GPU
        dev = cuda.Device(0)
        self.gpuName = dev.name()

        # conversion date
        timeNow = time.gmtime()
        self.conversionDate = time.strftime("%Y-%m-%d %H:%M:%S", timeNow)

    def getLineIgnoreComment(self, f, ignoreEmptyLines=True):
        """
        Read a line from the file f while ignoring commments.
        Comments start with '#'.
        """

        line = ""
        while True:
            # read line
            line = f.readline().decode("utf-8")

            # if end of file was reached -> break
            if not line:
                break

            # remove leading and trailing whitespace
            line = line.strip()

            # skip empty lines
            if ignoreEmptyLines and not line:
                continue

            # skip lines starting with '#'
            elif line.startswith("#"):
                continue

            break

        return line

    def readFromFile(self, f, raiseOnMissmatch=True):
        """
        Read header information from file
        """

        # header version
        version = int(self.getLineIgnoreComment(f))
        # always raise on header version missmatch
        if version != self.version:
            msg = "Version mismatch. File {}, expected {}".format(version, self.version)
            raise RuntimeError(msg)

        # gie/tensorrt version
        cudaVersion = int(self.getLineIgnoreComment(f))
        if raiseOnMissmatch and cudaVersion != self.cudaVersion:
            msg = "CUDA version mismatch. File {}, expected {}".format(cudaVersion, self.cudaVersion)
            raise RuntimeError(msg)
        self.cudaVersion = cudaVersion

        # cudnn version
        cudnnVersion = int(self.getLineIgnoreComment(f))
        if raiseOnMissmatch and cudnnVersion != self.cudnnVersion:
            msg = "cuDNN version mismatch. File {}, expected {}".format(cudnnVersion, self.cudnnVersion)
            raise RuntimeError(msg)
        self.cudnnVersion = cudnnVersion

        # gie/tensorrt version
        gieVersion = int(self.getLineIgnoreComment(f))
        if raiseOnMissmatch and gieVersion != self.gieVersion:
            msg = "Gie version mismatch. File {}, expected {}".format(gieVersion, self.gieVersion)
            raise RuntimeError(msg)
        self.gieVersion = gieVersion

        # GPU name
        self.gpuName = self.getLineIgnoreComment(f)

        # network name
        self.networkName = self.getLineIgnoreComment(f)

        # network name
        self.conversionDate = self.getLineIgnoreComment(f)

        # outputs with attributes
        nbOutputsWithAttributes = int(self.getLineIgnoreComment(f))
        for i in range(nbOutputsWithAttributes):
            blobName = self.getLineIgnoreComment(f)
            self.attributes[blobName] = self.getLineIgnoreComment(f)

        # argmax image layers
        nbArgmaxImageLayers = int(self.getLineIgnoreComment(f))
        for i in range(nbArgmaxImageLayers):
            blobName = self.getLineIgnoreComment(f)
            self.argmaxImageLayers[blobName] = self.getLineIgnoreComment(f)

        # mean values
        nbMeanVectors = int(self.getLineIgnoreComment(f))
        for i in range(nbMeanVectors):
            blobName = self.getLineIgnoreComment(f)
            meanVals = self.getLineIgnoreComment(f)
            self.meanValues[blobName] = [float(m) for m in meanVals.split(",") if m]

    def writeToFile(self, f):
        """
        Write header to file

        Arguments:
          f {[file]} -- file in binary encoding to write to
        """
        # write header version
        f.write("# version\n".encode("utf-8"))
        f.write("{}\n".format(self.version).encode("utf-8"))

        # write CUDA version
        f.write("# CUDA version\n".encode("utf-8"))
        f.write("{}\n".format(self.cudaVersion).encode("utf-8"))

        # write cudnn version
        f.write("# cudnn version\n".encode("utf-8"))
        f.write("{}\n".format(self.cudnnVersion).encode("utf-8"))

        # write tensorrt version
        f.write("# gie/tensorrt version\n".encode("utf-8"))
        f.write("{}\n".format(self.gieVersion).encode("utf-8"))

        # write gpu name
        f.write("# Converted with GPU\n".encode("utf-8"))
        f.write("{}\n".format(self.gpuName).encode("utf-8"))

        # network name
        f.write("# network name\n".encode("utf-8"))
        f.write("{}\n".format(self.networkName).encode("utf-8"))

        # conversion date
        f.write("# conversion date (UTC)\n".encode("utf-8"))
        f.write("{}\n".format(self.conversionDate).encode("utf-8"))

        # output buffers with atributes
        f.write("# number of output buffers with attributes\n".encode("utf-8"))
        f.write("{}\n".format(len(self.attributes)).encode("utf-8"))
        for i, attrName in enumerate(self.attributes.keys()):
            f.write("# output/attribute pair: {}\n".format(i).encode("utf-8"))
            f.write("{}\n".format(attrName).encode("utf-8"))
            f.write("{}\n".format(self.attributes[attrName]).encode("utf-8"))

        # argmax image layers
        f.write("# number of argmax image layers\n".encode("utf-8"))
        f.write("{}\n".format(len(self.argmaxImageLayers)).encode("utf-8"))
        for i, argmaxName in enumerate(self.argmaxImageLayers.keys()):
            f.write("# argmax image layer input/output pair: {}\n".format(i).encode("utf-8"))
            f.write("{}\n".format(argmaxName).encode("utf-8"))
            f.write("{}\n".format(self.argmaxImageLayers[argmaxName]).encode("utf-8"))

        # input buffers with mean values
        f.write("# number of input buffers with mean values\n".encode("utf-8"))
        f.write("{}\n".format(len(self.meanValues)).encode("utf-8"))
        for i, meanName in enumerate(self.meanValues.keys()):
            f.write("# mean input vector: {}\n".format(i).encode("utf-8"))
            f.write("{}\n".format(meanName).encode("utf-8"))
            meanVals = ["{:0.9f},".format(val) for val in self.meanValues[meanName]]
            f.write("{}\n".format("".join(meanVals)).encode("utf-8"))


class TestHeaders(unittest.TestCase):
    # for conditional tests, check for successful imports
    canRun = True
    try:
        import tensorrt
        import pycuda.driver
        import pycuda.autoinit
    except ImportError:
        canRun = False
        _logger.warning("Can't import necessary modules for tests. Not running any.")

    @unittest.skipUnless(canRun, "TensorRT or pycuda not available")
    def test_empty_header_init(self):
        # this should simply create a header file
        header = CGieHeader()  # noqa F841

    @unittest.skipUnless(canRun, "TensorRT or pycuda not available")
    def test_read_write(self):
        temp_file = tempfile.TemporaryFile()
        header = CGieHeader()

        # write header
        header.writeToFile(temp_file)
        # rewind file
        temp_file.seek(0)
        # read: should not be problem
        header.readFromFile(temp_file)
        temp_file.close()

    @unittest.skipUnless(canRun, "TensorRT or pycuda not available")
    def test_init_from_file_no_raise(self):
        fd, file_name = tempfile.mkstemp()
        header = CGieHeader()
        # make changes to header in every entry except version
        header.cudaVersion += 1
        header.cudnnVersion += 1
        header.gieVersion += 1
        header.gpuName = "lalala"
        header.networkName = "abcde"
        header.argmaxImageLayers = {"a": "b"}
        header.attributes = {"c": "d"}
        header.meanValues = {"e": [1.0, 2.0, 3.0]}

        # write header
        with open(file_name, "wb") as f:
            header.writeToFile(f)

        # read from file but allow missmatches
        new_header = CGieHeader(modelFile=file_name, raiseOnMissmatch=False)
        self.assertEqual(header.version, new_header.version)
        self.assertEqual(header.cudaVersion, new_header.cudaVersion)
        self.assertEqual(header.cudnnVersion, new_header.cudnnVersion)
        self.assertEqual(header.gieVersion, new_header.gieVersion)
        self.assertEqual(header.gpuName, new_header.gpuName)
        self.assertEqual(header.networkName, new_header.networkName)
        self.assertEqual(header.argmaxImageLayers, new_header.argmaxImageLayers)
        self.assertEqual(header.attributes, new_header.attributes)
        self.assertEqual(header.meanValues, new_header.meanValues)
        # clear files
        os.close(fd)
        os.remove(file_name)

    @unittest.skipUnless(canRun, "TensorRT or pycuda not available")
    def test_version_missmatch(self):
        temp_file = tempfile.TemporaryFile()
        header = CGieHeader()

        # write header
        header.writeToFile(temp_file)
        # rewind file
        temp_file.seek(0)
        # make header version missmatch
        header.version = header.version + 1
        with self.assertRaises(RuntimeError):
            # read: should not be problem
            header.readFromFile(temp_file)
        temp_file.close()

    @unittest.skipUnless(canRun, "TensorRT or pycuda not available")
    def test_cuda_missmatch(self):
        temp_file = tempfile.TemporaryFile()
        header = CGieHeader()

        # write header
        header.writeToFile(temp_file)
        # rewind file
        temp_file.seek(0)
        # make header version missmatch
        header.cudaVersion = header.cudaVersion + 1
        with self.assertRaises(RuntimeError):
            # read: should not be problem
            header.readFromFile(temp_file)
        temp_file.close()

    @unittest.skipUnless(canRun, "TensorRT or pycuda not available")
    def test_tensorrt_missmatch(self):
        temp_file = tempfile.TemporaryFile()
        header = CGieHeader()

        # write header
        header.writeToFile(temp_file)
        # rewind file
        temp_file.seek(0)
        # make header version missmatch
        header.gieVersion = header.gieVersion + 1
        with self.assertRaises(RuntimeError):
            # read: should not be problem
            header.readFromFile(temp_file)
        temp_file.close()

        temp_file = tempfile.TemporaryFile()
        header = CGieHeader()

        # write header
        header.writeToFile(temp_file)
        # rewind file
        temp_file.seek(0)
        # make header version missmatch
        header.cudaVersion = header.cudaVersion + 1
        with self.assertRaises(RuntimeError):
            # read: should not be problem
            header.readFromFile(temp_file)
        temp_file.close()


if __name__ == "__main__":
    unittest.main()
