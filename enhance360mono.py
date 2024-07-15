import numpy as np
import cv2
import os

def find_limits_mono(depth_image):
    # Assuming depth values increase with distance
    # You may need to invert this if your depth values decrease with distance
    valid_depths = depth_image[depth_image > 0]
    
    if len(valid_depths) == 0:
        return 0, 0, 0, 0, 0, 0

    # Use percentiles to estimate limits, ignoring potential outliers
    ceil_height = np.percentile(valid_depths, 5)
    floor_height = np.percentile(valid_depths, 95)
    
    h, w = depth_image.shape
    left_dist = 0
    right_dist = w - 1
    front_dist = np.percentile(valid_depths, 5)
    back_dist = np.percentile(valid_depths, 95)

    return ceil_height, floor_height, front_dist, back_dist, left_dist, right_dist

def fix_limits_mono(depth_image, ceil_height, floor_height, front_dist, back_dist, left_dist, right_dist):
    # Clip the depth values to the estimated range
    depth_image = np.clip(depth_image, front_dist, back_dist)
    
    # Normalize the depth values
    depth_range = back_dist - front_dist
    normalized_depth = (depth_image - front_dist) / depth_range
    
    # Scale back to the original depth range
    depth_image = front_dist + normalized_depth * depth_range
    
    return depth_image.astype(np.uint16)

def process(input_dir, depth_file, rgb_file, out_depth_file):
    depth_path = os.path.join(input_dir, depth_file)
    rgb_path = os.path.join(input_dir, rgb_file)
    out_path = os.path.join(input_dir, out_depth_file)

    print(f"Processing depth file: {depth_path}")
    print(f"Processing RGB file: {rgb_path}")
    print(f"Output will be saved to: {out_path}")

    # Read the depth image
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        print(f"Error: Could not read depth image from {depth_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Does the file exist? {os.path.exists(depth_path)}")
        print(f"Directory contents: {os.listdir(input_dir)}")
        return

    # Read the RGB image
    rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_image is None:
        print(f"Error: Could not read RGB image from {rgb_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Does the file exist? {os.path.exists(rgb_path)}")
        print(f"Directory contents: {os.listdir(input_dir)}")
        return

    print(f"Depth image shape: {depth_image.shape}, dtype: {depth_image.dtype}")
    print(f"RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")

    # Convert depth image to single channel if necessary
    if len(depth_image.shape) > 2:
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        print(f"Converted depth image to single channel. New shape: {depth_image.shape}")

    if depth_image.dtype != np.uint16:
        depth_image = (depth_image.astype(np.float32) * 65535 / 255).astype(np.uint16)
        print(f"Converted depth image to uint16. New dtype: {depth_image.dtype}")

    # Find room limits
    ceil_height, floor_height, front_dist, back_dist, left_dist, right_dist = find_limits_mono(depth_image)

    print(f"Room dimensions:")
    print(f"Height: {floor_height - ceil_height:.2f} units")
    print(f"Width: {right_dist - left_dist:.2f} units")
    print(f"Length: {back_dist - front_dist:.2f} units")

    # Fix limits
    fixed_depth = fix_limits_mono(
        depth_image, 
        ceil_height, floor_height, front_dist, back_dist, left_dist, right_dist
    )

    # Enhance contrast for visualization
    fixed_depth_normalized = cv2.normalize(fixed_depth, None, 0, 255, cv2.NORM_MINMAX)
    fixed_depth_visible = fixed_depth_normalized.astype(np.uint8)

    # Save the processed depth image
    cv2.imwrite(out_path, fixed_depth)
    print(f"Saved enhanced depth map to: {out_path}")

    # Save a visible version for easy inspection
    cv2.imwrite(out_path.replace('.png', '_visible.png'), fixed_depth_visible)
    print(f"Saved visible enhanced depth map to: {out_path.replace('.png', '_visible.png')}")

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="360 mono depth map enhancer")
    parser.add_argument("depth_map", help="Input depth map filename", type=str)
    parser.add_argument("rgb", help="Input RGB image filename", type=str)
    parser.add_argument("output", help="Output depth map filename", type=str)
    return parser.parse_args()

def main():
    args = parse_arguments()
    input_dir = os.path.join("Data", "Input")
    process(input_dir, args.depth_map, args.rgb, args.output)

if __name__ == '__main__':
    main()