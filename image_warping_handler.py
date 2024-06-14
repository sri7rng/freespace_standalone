
import os
import numpy as np
import json
import math

import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Tuple
#  warping & intrinsics/extrinsics
from scipy.spatial.transform import Rotation

from warper_package import image_warper
from transformation_lib_pybind import transformation_lib_pybind as tlp


CAM_OPENING_ANGLE = 195  # degree opening angle
CUT_ANGLE_LOWER = 20.0
CUT_ANGLE_UPPER = -20.0


class CExtrinsicDin70k(object):
    """Class for CMei extrinsic providing reading functionality from .txt file"""

    def __init__(self, P_cam2veh_din70k):

        P_cv2din70k = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        P_cam2veh_cv2din70k = np.dot(P_cam2veh_din70k, P_cv2din70k)

        self.P_cam2veh_din70k = P_cam2veh_din70k
        self.P_veh2cam_din70k = np.linalg.inv(P_cam2veh_din70k)
        self.P_cam2veh_cv2din70k = P_cam2veh_cv2din70k
        self.P_veh2cam_din70k2cv = np.linalg.inv(P_cam2veh_cv2din70k)
        self.P_cam_position = np.zeros((1, 3), dtype=float)

    @classmethod
    def fromLineIter(cls, lineIter):
        P_cam2veh_din70k = np.identity(4)
        for idx in range(3):
            P_cam2veh_din70k[idx, 0:4] = np.fromstring(next(lineIter), sep=" ")

        return cls(P_cam2veh_din70k)

    def getCamPositionFromPositionLine(self, lineIter):
        self.P_cam_position[:] = np.fromstring(next(lineIter).split(":")[-1], sep=" ")
        next(lineIter)  # euler angles (world2cam)
        next(lineIter)  # euler angles (cam2world)


def parse_parameters(path_intrinsic: str, extrinsic_in: Dict) -> Tuple[Dict, Dict]:
    """Parse extrinsic and intrinsic for a given camera

    Args:
        path_intrinsic (str): intrinsic path for this camera
        extrinsic_in (Dict): extrinsic data for this camera

    Returns:
        Tuple[Dict, Dict]: intrinsic and extrinsic value dicts
    """
    tree = ET.parse(path_intrinsic)
    root = tree.getroot()

    sm_cameras_omnicam = root.find("./sm_cameras_omnicam")
    sxy = float(sm_cameras_omnicam.find("./sxy").text)
    intrinsic = {}
    intrinsic["fx"] = float(sm_cameras_omnicam.find("./c").text)
    intrinsic["fy"] = intrinsic["fx"] * sxy
    intrinsic["cx"] = float(sm_cameras_omnicam.find("./xh").text)
    intrinsic["cy"] = float(sm_cameras_omnicam.find("./yh").text)
    intrinsic["r1"] = float(sm_cameras_omnicam.find("./A1").text)
    intrinsic["r2"] = float(sm_cameras_omnicam.find("./A2").text)
    intrinsic["r3"] = float(sm_cameras_omnicam.find("./A3").text)
    intrinsic["xi"] = float(sm_cameras_omnicam.find("./XI").text)

    rotation = Rotation.from_matrix(extrinsic_in.P_cam2veh_din70k[0:3, 0:3])
    quaternion = rotation.as_quat()
    extrinsic = {}
    extrinsic["quaternion"] = quaternion
    extrinsic["translation"] = extrinsic_in.P_cam2veh_din70k[:3, 3]

    return intrinsic, extrinsic

class ReadCameraParameter:

    def __init__(self, extrinsic_path, base_path):
        self.extrinsic_path = extrinsic_path
        self.base_path = base_path

        with open(self.extrinsic_path, "r") as fs:
            lines = (line.rstrip() for line in fs)
            lines = list(line for line in lines if line)

        lineIter = iter(lines)
        self.extrinsic = {}
        self.extrinsic["rear"] = CExtrinsicDin70k.fromLineIter(lineIter)
        self.extrinsic["left"] = CExtrinsicDin70k.fromLineIter(lineIter)
        self.extrinsic["front"] = CExtrinsicDin70k.fromLineIter(lineIter)
        self.extrinsic["right"] = CExtrinsicDin70k.fromLineIter(lineIter)



    def get_intrinsic_extrinsic(self, cam):
        intrinsic_cam, extrinsic_cam = parse_parameters(
                os.path.join(self.base_path, cam, "nrc_intrinsic_calib.xml"), self.extrinsic[cam]
            )

        return  intrinsic_cam, extrinsic_cam
    
    def write_warper_config(self, intrinsic: Dict, extrinsic: Dict, path_config: str) -> Dict:
        """Writes intrinsic and extrinsic data into dict for warper lib config & exports as json.

        Args:
            intrinsic (Dict): intrinsic data with fl&pp
            extrinsic (Dict): extrinsic data with rotation&translation
            path_config (str): config template path

        Returns:
            Dict: final config
        """
        with open(path_config, "r") as fp:
            config = json.load(fp)

        ## source
        config["source"]["intrinsics"]["focal_length"] = [intrinsic["fx"] / 2, intrinsic["fy"] / 2]
        config["source"]["intrinsics"]["principal_point"] = [intrinsic["cx"] / 2, intrinsic["cy"] / 2]
        config["source"]["intrinsics"]["radial_coefficients"] = [
            intrinsic["r1"],
            intrinsic["r2"],
            intrinsic["r3"],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        config["source"]["extrinsics"]["translation"] = extrinsic["translation"].tolist()
        config["source"]["extrinsics"]["rotation"] = extrinsic["quaternion"].tolist()
        with open(path_config, "w") as fp:
            json.dump(config, fp, indent=4)

        #return config
    
class GenerateImageWarper():

    def __init__(self, extrinsic_path, base_path):
        self.camera_parameters = ReadCameraParameter(extrinsic_path, base_path)
        self.config_path = "config_example_cmei_deformedcylinder.json"
        self.cam_warper = None
        self.intrinsic_cam = None
        #self.extrinsic_cam = None

    def create_warper(self,cam):
        config_path_cam = self.config_path.replace(".json", f"_{cam}.json")
        self.intrinsic_cam, extrinsic_cam = self.camera_parameters.get_intrinsic_extrinsic(cam)
        self.camera_parameters.write_warper_config(self.intrinsic_cam, extrinsic_cam,config_path_cam)
        self.cam_warper = image_warper.ImageWarper(config_path_cam)
    
    def warp_image(self,img):
        img_warped = self.cam_warper.warp_image(img) # [H, W, C]
        img_warped = np.expand_dims(np.transpose(img_warped, (2, 0, 1)), 0)  # [Batch, C, H, W]

        # C-order for arrays needed because otherwise causes TRT inference issues
        img_warped = np.array(img_warped, order="C", dtype=np.float32)

        return img_warped
    
    def create_pixel_lightray_lut(self, height, width, cam):#, focal_length, principal_point, cam):
        focal_length = [self.intrinsic_cam["fx"] / 2, self.intrinsic_cam["fy"] / 2]
        principal_point = [self.intrinsic_cam["cx"] / 2, self.intrinsic_cam["cy"] / 2]
        cam_rotations = {"front": 0.0, "left": math.pi / 2, "rear": math.pi, "right": -math.pi / 2}
        cam_rot_rad = cam_rotations[cam]
        lut = np.full((height, width, 3), np.nan)
        rot_cv_to_din70k = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        rot_cam_to_veh = np.array(
            [
                [math.cos(cam_rot_rad), -math.sin(cam_rot_rad), 0.0],
                [math.sin(cam_rot_rad), math.cos(cam_rot_rad), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        transformation_lib = tlp.viper.transformation_lib
        viper = tlp.viper

        intrinsic_data = tlp.viper.intrinsic_mock_values.makeNominalIntrinsicDeformedCylinder()
        intrinsic_data.image_dimensions = tlp.vfc.linalg.TVector2_int_t(height, width)
        intrinsic_data.focal_length = tlp.vfc.linalg.TVector2_float_t(*focal_length)
        intrinsic_data.principal_point = tlp.vfc.linalg.TVector2_float_t(*principal_point)
        intrinsic_data.cut_angle_lower = tlp.vfc.TDegree_float_t(CUT_ANGLE_LOWER)
        intrinsic_data.cut_angle_upper = tlp.vfc.TDegree_float_t(CUT_ANGLE_UPPER)

        transformation_instance = transformation_lib.createTransformation(intrinsic_data)

        for u in range(width):
            for v in range(height):
                # Pixel to light ray via deformed cylinder rules
                light_ray_object = transformation_instance.pixelToLightRaySensor(tlp.vfc.linalg.TVector2_float_t(u, v))

                light_ray = np.array(
                    [light_ray_object.value().x(), light_ray_object.value().y(), light_ray_object.value().z()]
                )

                # convert CV coordinate system to DIN70k by switchting axis
                # light_ray_din70k = np.dot(rot_cv_to_din70k, light_ray)

                # convert cam to vehicle coordinates by rotations around Z-axis
                # position is still in camera
                light_ray_veh_din70k = np.dot(rot_cam_to_veh, light_ray)

                lut[u, v] = light_ray_veh_din70k
        return lut

