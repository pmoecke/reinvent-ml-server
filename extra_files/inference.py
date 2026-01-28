
import argparse
import csv
import json
import math
from efm3d.inference.pipeline2 import run_one
from vrs_to_calibrated import calibrate
import os

class CSV_to_WEB:
    
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    def quaternion_to_euler(qw, qx, qy, qz):
        
        w,x,y,z = qw, qx, qy, qz

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x*x + y*y)
        X = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = CSV_to_WEB.clamp(t2, -1.0, 1.0)
        Y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y*y + z*z)
        Z = math.atan2(t3, t4)

        return X, Y, Z

    def csv_to_three_boxes(csv_path, output_path, color="#8B0000"):
        boxes = []
        id = 0
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tx = float(row.get('tx_world_object',0.0))
                ty = float(row.get('ty_world_object',0.0))
                tz = float(row.get('tz_world_object',0.0))
                qw = float(row.get('qw_world_object',0.0))
                qx = float(row.get('qx_world_object',0.0))
                qy = float(row.get('qy_world_object',0.0))
                qz = float(row.get('qz_world_object',0.0))
                sx = float(row.get('scale_x',0.0))
                sy = float(row.get('scale_y',0.0))
                sz = float(row.get('scale_z',0.0))
                pr = float(row.get('prob',0.0))
                label = row.get('name', 'NO_NAME')

                ex, ey, ez = CSV_to_WEB.quaternion_to_euler(qw, qx, qy, qz)

                box = {
                    "id": id,
                    "cmd": "",
                    "label": label,
                    "center": [tx, ty, tz],
                    "euler": [ex, ey, ez],
                    "scale": [sx, sy, sz],
                    "color": color,
                    "prob": pr
                }
                boxes.append(box)
                id += 1
        
        json_str = json.dumps(boxes)
        with open(os.path.join(output_path, 'web_boxes.json'),"w") as f:
            f.write(json_str)

def run_efm_inference():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="input data"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output dir"
    )
    
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="./ckpt/model_lite.pth",
        help="model checkpoint path",
    )
    
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="./efm3d/config/evl_inf_desktop.yaml",
        help="model config file",
    )
    
    parser.add_argument(
        "--num_snips",
        type=int,
        default=100_000,
        help="number of snippets per sequence, by default evaluate the full sequence",
    )
    
    parser.add_argument(
        "--snip_stride",
        type=float,
        default=0.1,
        help="overlap between snippets in second, default to 0.1 (recommend to set it between 0.1-0.5), set it larger will make performance worse but run faster",
    )
    
    args = parser.parse_args()
    
    vrs_path = args.input
    
    if not args.output_dir:
        output_dir = os.path.dirname(vrs_path)
    else:
        output_dir = args.output_dir
    
    calibrated_output_dir = os.path.join(output_dir, 'calibrated')
    inference_output_dir = os.path.join(output_dir, 'inference')
    
    if not os.path.exists(calibrated_output_dir):
        os.makedirs(calibrated_output_dir, exist_ok=True)
        
    if not os.path.exists(inference_output_dir):
        os.makedirs(inference_output_dir, exist_ok=True)
    
    csv_path = os.path.join(inference_output_dir, 'tracked_scene_obbs.csv')
    
    
    calibrate(
        input_path=os.path.dirname(vrs_path), 
        output_path=calibrated_output_dir
    )
    
    run_one(
        data_path=vrs_path,
        model_ckpt=args.model_ckpt,
        model_cfg=args.model_cfg,
        max_snip=args.num_snips,
        snip_stride=args.snip_stride,
        output_dir=inference_output_dir,
    )
    
    CSV_to_WEB.csv_to_three_boxes(csv_path, inference_output_dir)
    
    
    

if __name__ == '__main__':
    run_efm_inference()