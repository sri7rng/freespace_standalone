import os
import argparse
import numpy as np
from image_warping_handler import GenerateImageWarper
from rof_handler import read_rof_file, write_rof
from cuda_handler import load_cuda_engine
from stixel_handler import plot_image, convert_disparity, convert_predict_format
from utils.inference import run_inference
from PIL import Image
import json


STIXEL_WIDTH = 8  # 8px per stixel
OLD_MAX_DEPTH = 100.0
CAM_OPENING_ANGLE = 195  # degree opening angle
CUT_ANGLE_LOWER = 20.0
CUT_ANGLE_UPPER = -20.0

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

def inference_on_each_image(base_path, rof_file, extrinsic_path, trt_model_path, create_video=False):
    cameras = ["front", "left", "rear", "right"]
    show_pred = False
    do_disparity_conv = True
    batch_size = 1000
    MAX_NR_PREDS = np.inf

    # create TRT inference engine
    cuda_engine, metadata = load_cuda_engine(trt_model_path)#, trt.Logger.ERROR)
    image_warper = GenerateImageWarper(extrinsic_path, base_path)
    
    for cam in cameras:
        run_inference_loop = True
        pred_path = os.path.join("/app", "prediction", cam)
        os.makedirs(pred_path, exist_ok=True)
        image_warper.create_warper(cam)

        img_batch = 0
        all_preds = []
        while run_inference_loop:
            img_list,timestamps = read_rof_file(rof_file, img_batch, batch_size, cam)
            len_timestamps = len(timestamps)

            for img_idx, img in enumerate(img_list):
                stixel_width_rad = (CAM_OPENING_ANGLE / img.shape[1] * STIXEL_WIDTH) * (np.pi / 180.0)
                img_warped = image_warper.warp_image(img)
                predictions = run_inference(cuda_engine, [img_warped])
                pred_output_file = {
                        "stixel_tv_y_bottom": predictions["stixel_tv_y_bottom"].tolist(),
                        "stixel_tv_y_top": predictions["stixel_tv_y_top"].tolist(),
                        "stixel_tv_class_id": predictions["stixel_tv_class_id"].tolist(),
                        "stixel_tv_disparity": predictions["stixel_tv_disparity"].tolist(),
                    }

                if do_disparity_conv:
                    pred_output_file = convert_disparity(pred_output_file)

                pred_output_file["stixel_width_rad"] = float(stixel_width_rad)
                pred_output_file["timestamp"] = int(timestamps[img_idx+img_batch])
                all_preds.append(pred_output_file)

                # plot every image
                if create_video:
                    img_nr_padded = str(img_batch+img_idx).zfill(8)
                    img_visu_stixel, img_visu_depth = plot_image(img_warped, pred_output_file)
                    Image.fromarray(img_visu_stixel.astype(np.uint8)).save(
                        os.path.join(pred_path, f"{img_nr_padded}_{cam}_image_stixels_{str(timestamps[img_idx+img_batch])}.jpg")
                        )
                    Image.fromarray(img_visu_depth.astype(np.uint8)).save(
                        os.path.join(pred_path, f"{img_nr_padded}_{cam}_image_depth_{str(timestamps[img_idx+img_batch])}.jpg")
                        )
            

            # Update loop condition based on returned rof size
            img_batch += batch_size
            if img_batch >= len_timestamps or img_batch >= MAX_NR_PREDS:
                run_inference_loop = False
        
        # TODO: make rof reading work directly inside pipeline
        # write_rof(all_preds, path=rof_file.replace("Recording", "Stixel_Prediction"), cam=cam)
        with open(os.path.join(pred_path, "all_preds.json"), "w") as fp:
            json.dump(all_preds, fp, cls=NumpyEncoder)

        #t_cam_end = time.perf_counter()
        #logging.info(f"{'CAM:' + cam:<10} Total time: {t_cam_end-t_cam_start}")

        # Create video
        if create_video:
            os.system(f"ffmpeg -y -r 15 -pattern_type glob -i '{pred_path}/*_stixels_*.jpg' -crf 24 -c:v libx264 -preset veryfast -pix_fmt yuv420p /app/prediction/{cam}_pred.mp4")

def save_inferences_on_rof_file(base_path, rof_file, extrinsic_path):
    cameras = ["left", "front", "rear", "right"]
    image_size = (768, 768)  # height, width

    image_warper = GenerateImageWarper(extrinsic_path, base_path)

    for cam in cameras:
        image_warper.create_warper(cam)

        lut = image_warper.create_pixel_lightray_lut(image_size[0],
            image_size[1],
            cam)
        
        pred_path = os.path.join("/app", "prediction", cam)
        os.makedirs(pred_path, exist_ok=True)

        with open(os.path.join(pred_path, "all_preds.json"), "r") as fp:
            all_preds = json.load(fp)

        for idx, pred_dict in enumerate(all_preds):
            pred_new_format = {}
            for k, v in pred_dict.items():
                pred_new_format[k] = np.squeeze(np.asarray(v))
            all_preds[idx] = convert_predict_format(pred_new_format, lut)
            all_preds[idx]["timestamp"] = pred_new_format["timestamp"]
            all_preds[idx]["stixel_width_rad"] = pred_new_format["stixel_width_rad"]

        write_rof(all_preds, path="/app/prediction/20240222T092940Z_LBXO1994_C1_DEV_ARGUS_LiDAR_MTA2_0_Stixel_Prediction.rof", cam=cam)

        #t_cam_end = time.perf_counter()
        #logging.info(f"{'CAM:' + cam:<10} Total time: {t_cam_end-t_cam_start}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daq2.0 inference chain with PACE stixelnet")
    parser.add_argument(
        "--extrinsic_path",
        type=str,
        help="Path to extrinsic.",
        required=False,
        default="data/mf4_conversion/new_set/755584/extrinsic/extrinsic_din70k.txt",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        help="Base folder path where prediction folder is created and per camera intrinsic folders are",
        required=False,
        default="data/mf4_conversion/new_set/755584",
    )
    parser.add_argument(
        "--trt_model_path",
        type=str,
        help="Path to trt model .gie file",
        required=False,
        default="data/mf4_conversion/new_set/model_inference_768x768_NVIDIA_GeForce_RTX_3090.gie",
    )
    parser.add_argument(
        "--rof_file",
        type=str,
        help="Path to rof file",
        required=False,
        default="data/mf4_conversion/new_set/755584/20240222T092940Z_LBXO1994_C1_DEV_ARGUS_LiDAR_MTA2_0_Recording.rof",
    )

    args = parser.parse_args()
    inference_on_each_image(args.base_path, args.rof_file, args.extrinsic_path, args.trt_model_path)
    save_inferences_on_rof_file(args.base_path, args.rof_file, args.extrinsic_path)

