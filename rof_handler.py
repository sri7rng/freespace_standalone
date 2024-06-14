
import time
import numpy as np
import json
import os
import struct

# Julia
from julia import Julia

_ = Julia(compiled_modules=False)
from julia import Main
from julia import GenericRofReader

# TODO: include julia file here from other repo
print("loading julia")
Main.include("/app/camera.jl")
print("julia loaded")


def read_rof_file(rof_file, img_batch, batch_size, cam):
    # rof reading
    print("starting rof reading")
    t_rof_start = time.perf_counter()
    img_list, timestamps = Main.convert_rof_to_camera_data(rof_file, img_batch, batch_size, cam)
    len_timestamps = len(timestamps)
    t_rof_end = time.perf_counter()
    print(f"{'CAM:' + cam:<10} Time Rof reading: {t_rof_end - t_rof_start}")
    print(f"{'CAM:' + cam:<10} run warping&inference on {len(img_list)} images with {len(timestamps)} timestamps")
    
    return img_list, timestamps

def write_rof(all_predictions, path, cam):
    source_id = "stixels_" + cam
    stixel_hash = "9000644979cbab5fbeb92271536dd190"
    rof_type_hashes = {
        "JsonVersion": "v1.0.0",
        stixel_hash: {
            "ClassProperties": {"Align": 4, "AlignAs": 1, "Attributes": "top-class has-members ", "Size": 28},
            "StructTypeClassHash": stixel_hash,
            "Members": {
                "m_azimuth": {"Offset": 20, "RfMTFlags": "--", "Type": "float", "Count": 1, "TypeSize": 4},
                "m_depth": {"Offset": 16, "RfMTFlags": "--", "Type": "float", "Count": 1, "TypeSize": 4},
                "InheritedType": "null",
                "StructType": "rbp::nfm::debug::CStixelType",
                "m_class_id": {"Offset": 24, "RfMTFlags": "--", "Type": "uint32_t", "Count": 1, "TypeSize": 4},
                "m_timestamp": {"Offset": 0, "RfMTFlags": "--", "Type": "uint32_t", "Count": 1, "TypeSize": 4},
                "m_h_bottom": {"Offset": 12, "RfMTFlags": "--", "Type": "float", "Count": 1, "TypeSize": 4},
                "m_stixel_width": {"Offset": 4, "RfMTFlags": "--", "Type": "float", "Count": 1, "TypeSize": 4},
                "Count": 0,
                "StructTypeClassHash": stixel_hash,
                "m_h_top": {"Offset": 8, "RfMTFlags": "--", "Type": "float", "Count": 1, "TypeSize": 4},
            },
            "StructType": "rbp::nfm::debug::CStixelType",
        },
        "abiFlagHex32Bit": "00291c56",
    }
    rof_type_hashes_filename = "typeHash_" + stixel_hash + "_abiFlagsHex32Bit_00291c56"
    sub_level_struct_size = {
        "e14dff9373323e56ea8d7098076942f0": 12,
        "bd1f3b95c28b44f8a299aae0b6da2d58": 376,
        "3fdf76826e77f9713fd1fd6abe2f7108": 8,
        "47da705698ea91e2d4d908345b163a08": 376004,
        "603a954bc9db442518c652dca31a5620": 1,
        "5b38ce521ae7d3fd076ebb0f2e356188": 8,
        "JsonVersion": "v1.0.0",
        "92c3a9314a919908c77573236bbdb940": 8,
        "abbb54f544362f70d389c4dae8064bf0": 8,
        stixel_hash: 28,
        "abiFlagHex32Bit": "00291c56",
        "891e80b980b3c35f6580495aad4b4c00": 8,
        "b148f7fff92000d52686fc67b93ff6e0": 364,
        "d572b2349223fe3d649846a6e3c366b0": 376016,
        "34e790aaff63fe3e21a2748bcb2c3ee0": 376000,
        "caf3fe785cc2e4f67faa2d6120938128": 364,
        "8cf14acf191ae7b5dfc471b7175fa7c0": 8,
        "1249f414cf97a754248317895655da78": 8,
    }
    stixel_meta = {
        "payloadSize": 28,
        "uniqueSourceID": source_id,
        "typeHash": "9000644979cbab5fbeb92271536dd190",
        "MTA_info": {"num_missing_mtas": 0, "padding_byte": "0x69", "missing_mtas_idx_list": []},
        "abiFlagsHex32Bit": "0x291c56",
    }
    os.makedirs(path, exist_ok=True)

    # write ROF interface data & interface meta
    with open(os.path.join(path, source_id + "_meta.json"), "w") as fp:
        json.dump(stixel_meta, fp)
    with open(os.path.join(path, source_id + "_data"), "wb") as fp:
        for predictions in all_predictions:
            for stixel_idx in range(predictions["stixel_tv_h_bottom"].shape[0]):
                data_bytes = struct.pack(
                    "IfffffI",
                    predictions["timestamp"],
                    predictions["stixel_width_rad"],
                    predictions["stixel_tv_h_top"][stixel_idx],
                    predictions["stixel_tv_h_bottom"][stixel_idx],
                    predictions["stixel_tv_depth"][stixel_idx],
                    predictions["stixel_tv_azimuth"][stixel_idx],
                    int(predictions["stixel_tv_class_id"][stixel_idx]),
                )

                # Write the bytes to the file
                fp.write(data_bytes)

    # write ROF class info files if not existent
    class_info_dir = os.path.join(path, "class_info_jsons_v1-0-0")
    os.makedirs(class_info_dir, exist_ok=True)
    rof_type_hashes_path = os.path.join(class_info_dir, rof_type_hashes_filename + ".json")
    if not os.path.exists(rof_type_hashes_path):
        with open(rof_type_hashes_path, "w") as fp:
            json.dump(rof_type_hashes, fp)

    sub_level_structs_path = os.path.join(class_info_dir, "sub_level_struct_sizes" + ".json")
    if not os.path.exists(sub_level_structs_path):
        with open(sub_level_structs_path, "w") as fp:
            json.dump(sub_level_struct_size, fp)

    return

