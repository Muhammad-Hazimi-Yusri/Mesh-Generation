import argparse
import numpy as np
from lib_edgenet360.py_cuda import lib_edgenet360_setup, get_point_cloud
import cv2
import matplotlib.pyplot as plt
from lib_edgenet360.preproc import find_limits_v2, find_planes, fix_limits_mono
import os

PI = 3.14159265
DISP_SCALE = 2.0
DISP_OFFSET = -120.0

BASELINE = 0.264
V_UNIT = 0.02
DATA_PATH = './Data'

def visualize_step(image, title, filename):
    plt.figure(figsize=(10, 5))
    plt.imshow(image, cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.savefig(filename)
    plt.close()

def process(depth_file, rgb_file, out_depth_file, baseline):
    print(f"Processing depth file: {depth_file}")
    print(f"Processing RGB file: {rgb_file}")

    lib_edgenet360_setup(device=0, num_threads=1024, v_unit=V_UNIT, v_margin=0.24, f=518.8579, debug=0)

    point_cloud, depth_image = get_point_cloud(depth_file, baseline=baseline)
    visualize_step(depth_image, 'Initial Depth Image', 'debug_step1_initial_depth.png')
    print(f"Initial depth range: {depth_image.min()} - {depth_image.max()}")

    bgr_image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)

    edges_image = cv2.Canny(bgr_image, 30, 70)
    kernel = np.ones((5,5), np.uint8)
    thin_edges = cv2.dilate(edges_image, kernel, iterations=1)
    wide_edges = cv2.dilate(edges_image, kernel, iterations=3)

    bilateral = cv2.bilateralFilter(bgr_image, 3, 75, 75)

    new_depth_image, region_mask, edges_mask, inf_region_mask, close_region_mask = find_planes(
        point_cloud, bilateral, wide_edges, depth_image, thin_edges, baseline=baseline
    )
    visualize_step(new_depth_image, 'Depth After Find Planes', 'debug_step2_after_find_planes.png')
    print(f"Depth after find_planes range: {new_depth_image.min()} - {new_depth_image.max()}")

    ceil_height, floor_height, front_dist, back_dist, right_dist, left_dist = find_limits_v2(point_cloud)

    print(f"Room dimensions:")
    print(f"Height: {ceil_height - floor_height:.2f} ({ceil_height:.2f} <> {floor_height:.2f})")
    print(f"Width: {right_dist - left_dist:.2f} ({right_dist:.2f} <> {left_dist:.2f})")
    print(f"Length: {front_dist - back_dist:.2f} ({front_dist:.2f} <> {back_dist:.2f})")

    fixed_depth = fix_limits_mono(new_depth_image,
                                  ceil_height, floor_height, front_dist, back_dist, right_dist, left_dist)
    visualize_step(fixed_depth, 'Final Enhanced Depth', 'debug_step3_final_enhanced_depth.png')
    print(f"Fixed depth range: {fixed_depth.min()} - {fixed_depth.max()}")

    # Additional visualization: difference between original and enhanced
    depth_diff = fixed_depth.astype(np.float32) - new_depth_image.astype(np.float32)
    visualize_step(depth_diff, 'Depth Difference (Enhanced - Original)', 'debug_step4_depth_difference.png')

    # Ensure the depth map is in the correct range for saving
    fixed_depth_normalized = cv2.normalize(fixed_depth, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    
    cv2.imwrite(out_depth_file, fixed_depth_normalized)
    print(f"Saved enhanced depth map to: {out_depth_file}")
    
    # Double-check the saved image
    saved_image = cv2.imread(out_depth_file, cv2.IMREAD_UNCHANGED)
    if saved_image is not None:
        print(f"Saved image range: {saved_image.min()} - {saved_image.max()}")
        visualize_step(saved_image, 'Saved Enhanced Depth', 'debug_step5_saved_enhanced_depth.png')
    else:
        print("Error: Failed to read the saved image.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="360 depth maps enhancer (Debug Version)")
    parser.add_argument("dataset", help="360 dataset dir", type=str)
    parser.add_argument("depth_map", help="360 depth map", type=str)
    parser.add_argument("rgb", help="360 rgb", type=str)
    parser.add_argument("output", help="output file prefix", type=str)
    parser.add_argument("--baseline", help="Stereo 360 camera baseline. Default 0.264", type=float, default=0.264, required=False)
    parser.add_argument("--data_path", help="Data path. Default ./Data", type=str, default="./Data", required=False)

    args = parser.parse_args()
    
    depth_map = os.path.join(args.data_path, args.dataset, args.depth_map)
    rgb_file = os.path.join(args.data_path, args.dataset, args.rgb)
    output = os.path.join(args.data_path, args.dataset, args.output)
    
    return depth_map, rgb_file, output, args.baseline

def main():
    depth_map, rgb_file, output, baseline = parse_arguments()
    process(depth_map, rgb_file, output, baseline)

if __name__ == '__main__':
    main()