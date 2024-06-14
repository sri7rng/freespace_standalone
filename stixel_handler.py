import numpy as np
from typing import Any, Callable, Dict, List, Tuple
import cv2
import os
import math


CLASS_TO_COLOR = {
    0: (255, 0, 127),
    1: (0, 0, 142),
    2: (220, 220, 0),
    3: (0, 255, 255),
}


STIXEL_WIDTH = 8  # 8px per stixel
OLD_MAX_DEPTH = 100.0
CAM_OPENING_ANGLE = 195  # degree opening angle
CUT_ANGLE_LOWER = 20.0
CUT_ANGLE_UPPER = -20.0
MIN_DEPTH = 0.3

def convert_disparity(task_output: Dict, EPSILON: float = 1e-24, raw_disparity: bool = False) -> Dict:
    """Convert disparity values to depth

    Args:
        task_output (Dict): stixelnet output dictionary
        EPSILON (float, optional): Float offset. Defaults to 1e-24.
        raw_disparity (bool, optional): toggle either raw or sigmoid disparity. Defaults to False.

    Returns:
        Dict: _description_
    """
    if raw_disparity:
        depth = 1.0 / (task_output["stixel_tv_disparity"] + EPSILON)
    else:
        # apply sigmoid function
        disparity_activation = 1.0 / (1.0 + np.exp(-np.array(task_output["stixel_tv_disparity"])))
        depth = depth_from_disparity(disparity_activation, OLD_MAX_DEPTH)

    return {
        "stixel_tv_y_bottom": task_output["stixel_tv_y_bottom"],
        "stixel_tv_y_top": task_output["stixel_tv_y_top"],
        "stixel_tv_class_id": task_output["stixel_tv_class_id"],
        "stixel_tv_depth": depth.tolist(),
    }


def visualize_stixel(image: np.ndarray, pred_output_file: Dict):
    """Draw stixels into image.

    Args:
        image: RGB image; in NHWC format
        pred_output_file: the resulted stixel_freespace
    Returns:
        image as tensor with freespace drawn inside the image
    """

    # height, width, channels
    img_height, img_width, _ = image.shape
    # batch_size, num_stixels
    num_stixels = len(pred_output_file["stixel_tv_y_bottom"][0])
    stixel_width = int(img_width // num_stixels)

    colors = [(0, 255, 0), (255, 0, 0)]
    image = image.astype("uint8")

    # plot predicted stixels on the image
    for column_idx, (y_bottom, y_top, class_id) in enumerate(
        zip(
            pred_output_file["stixel_tv_y_bottom"][0],
            pred_output_file["stixel_tv_y_top"][0],
            pred_output_file["stixel_tv_class_id"][0],
        )
    ):
        if not np.any(np.isnan(y_bottom)):
            point_1 = (
                int(column_idx * stixel_width),
                int(np.clip(y_bottom, 0, img_height - 1)),
            )
            point_2 = (
                int((column_idx + 1) * stixel_width),
                int(np.clip(y_top, 0, img_height - 1)),
            )
            image = cv2.rectangle(
                img=image,
                pt1=point_1,
                pt2=point_2,
                color=CLASS_TO_COLOR[int(class_id)],
                thickness=2,
            )
    return image


def depth_from_disparity(
    disparity: np.ndarray, max_depth: float, min_depth: float = MIN_DEPTH, epsilon: float = 1e-24
) -> np.ndarray:
    """The function convert disparity output from network to the depth.

    This function could potentially be moved to a common place once that exists so that other usecases can use it.

    Args:
        disparity: the network output (the sigmoid output should be between 0 and 1)
        min_depth: the minimum depth in meters
        max_depth: the maximum depth in meters
        epsilon: small float for numerical stability

    Returns:
        depth: the  depth in meters
    """
    # moved to postprocessing
    max_disparity = 1.0 / min_depth
    min_disparity = 1.0 / max_depth
    scaled_disparity = (max_disparity - min_disparity) * disparity + min_disparity
    depth = 1.0 / (scaled_disparity + epsilon)
    return depth

def convert_disparity(task_output: Dict, EPSILON: float = 1e-24, raw_disparity: bool = False) -> Dict:
    """Convert disparity values to depth

    Args:
        task_output (Dict): stixelnet output dictionary
        EPSILON (float, optional): Float offset. Defaults to 1e-24.
        raw_disparity (bool, optional): toggle either raw or sigmoid disparity. Defaults to False.

    Returns:
        Dict: _description_
    """
    if raw_disparity:
        depth = 1.0 / (task_output["stixel_tv_disparity"] + EPSILON)
    else:
        # apply sigmoid function
        disparity_activation = 1.0 / (1.0 + np.exp(-np.array(task_output["stixel_tv_disparity"])))
        depth = depth_from_disparity(disparity_activation, OLD_MAX_DEPTH)

    return {
        "stixel_tv_y_bottom": task_output["stixel_tv_y_bottom"],
        "stixel_tv_y_top": task_output["stixel_tv_y_top"],
        "stixel_tv_class_id": task_output["stixel_tv_class_id"],
        "stixel_tv_depth": depth.tolist(),
    }

def draw_depth_line(
    image: np.ndarray,
    pred_output_file: Dict,
    column_width: int = STIXEL_WIDTH,
):
    """Draw 2d depth plot

    Args:
        image (np.ndarray): rgb input image [H, W, C]
        pred_output_file (Dict): prediction results for the image
        column_width (int, optional): _description_. Defaults to 8.

    Returns:
        np.ndarray: depth plot array
    """
    MAX_DEPTH_VALUE = 150
    height, width, _ = image.shape
    depth_line_plot = np.zeros(shape=image.shape, dtype=np.uint8)
    depth_line_plot.fill(255)
    for col, (y_bottom, y_top, class_id, depth) in enumerate(
        zip(
            pred_output_file["stixel_tv_y_bottom"][0],
            pred_output_file["stixel_tv_y_top"][0],
            pred_output_file["stixel_tv_class_id"][0],
            pred_output_file["stixel_tv_depth"][0],
        )
    ):
        if y_bottom < 0 or y_top < 0:
            continue
        y_bottom = min(y_bottom, height - 1)

        right_edge = col * column_width + column_width
        right_edge = min(right_edge, width)

        rgb_color = tuple(CLASS_TO_COLOR[int(class_id)])
        thickness = 2

        # visualize depth
        if depth is not None and depth >= 0 and not np.isnan(depth) and not np.isinf(depth):
            y_depth = int((1 - depth / MAX_DEPTH_VALUE) * height)
            depth_line_plot = cv2.line(
                depth_line_plot, (col * column_width, y_depth), (right_edge, y_depth), rgb_color, thickness
            )
    return depth_line_plot


def plot_image(img_warped, pred_output_file):
    img_warped = np.transpose(img_warped[0], (1, 2, 0))  # [H, W, C]
    img_warped = np.ascontiguousarray(img_warped, dtype=np.uint8)
    img_visu_stixel = visualize_stixel(img_warped, pred_output_file)
    img_visu_depth = draw_depth_line(img_warped, pred_output_file)
    
    return img_visu_stixel, img_visu_depth

def convert_predict_format(predictions, lut):
    new_preds = {
        "stixel_tv_h_top": np.full_like(predictions["stixel_tv_y_top"], np.nan),
        "stixel_tv_h_bottom": np.full_like(predictions["stixel_tv_y_bottom"], np.nan),
        "stixel_tv_depth": np.asarray(predictions["stixel_tv_depth"]),
        "stixel_tv_azimuth": np.full_like(predictions["stixel_tv_y_top"], np.nan),
        "stixel_tv_class_id": np.asarray(predictions["stixel_tv_class_id"]),
    }
    for stixel_idx in range(predictions["stixel_tv_y_bottom"].shape[0]):
        y_bottom = predictions["stixel_tv_y_bottom"][stixel_idx]
        y_top = predictions["stixel_tv_y_top"][stixel_idx]
        radial_depth = predictions["stixel_tv_depth"][stixel_idx]  # in m

        ray_y_bottom = lut[stixel_idx * STIXEL_WIDTH, int(y_bottom)]
        ray_y_top = lut[stixel_idx * STIXEL_WIDTH, int(y_top)]

        azimuth = math.atan2(ray_y_bottom[1], ray_y_bottom[0])  # rad
        elevation_bottom = math.atan2(ray_y_bottom[2], np.hypot(ray_y_bottom[0], ray_y_bottom[1]))  # rad
        elevation_top = math.atan2(ray_y_top[2], np.hypot(ray_y_top[0], ray_y_top[1]))  # rad

        euclidian_depth_bottom = radial_depth / math.cos(elevation_bottom)
        euclidian_depth_top = radial_depth / math.cos(elevation_top)

        # TODO check if scaling with euclidian is correct
        stixel_bottom_din70k = (ray_y_bottom / np.linalg.norm(ray_y_bottom)) * euclidian_depth_bottom
        stixel_top_din70k = (ray_y_top / np.linalg.norm(ray_y_top)) * euclidian_depth_top

        new_preds["stixel_tv_h_top"][stixel_idx] = stixel_top_din70k[2]  # din70k z is height
        new_preds["stixel_tv_h_bottom"][stixel_idx] = stixel_bottom_din70k[2]  # din70k z is height
        new_preds["stixel_tv_azimuth"][stixel_idx] = azimuth
    return new_preds