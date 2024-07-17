import numpy as np
import cv2
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt


def find_planes_mono(depth_image, rgb_image, edges_image, thin_edges, visualize=False):
    h, w = depth_image.shape
    complete_region_mask = np.zeros((h, w), np.uint8)
    inf_region_mask = np.zeros((h, w), np.uint8)
    close_region_mask = np.zeros((h, w), np.uint8)
    new_depth_image = depth_image.copy()

    step = 75
    combined = rgb_image.copy()

    if visualize:
        plt.figure(figsize=(12, 6))

    for startx in np.arange(0, w, step):
        for starty in np.arange(250, h-250, step):
            if (inf_region_mask[starty, startx] > 0) or \
               (close_region_mask[starty, startx] > 0) or \
               (complete_region_mask[starty, startx] > 0):
                continue

            region_mask = find_region(startx, starty, rgb_image, step, thin_edges)
            edges_mask = region_mask & edges_image

            region_depths = depth_image[region_mask > 0]
            if len(region_depths) == 0:
                continue

            median_depth = np.median(region_depths)
            depth_std = np.std(region_depths)

            if depth_std < 500:  # Adjust this threshold as needed
                new_depth = median_depth * np.ones_like(region_depths)
            else:
                # Fit a plane to the depth values
                y, x = np.where(region_mask > 0)
                A = np.column_stack((x, y, np.ones_like(x)))
                b = region_depths
                plane_params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                
                # Calculate new depth values based on the fitted plane
                new_depth = np.dot(A, plane_params)

            if np.max(new_depth) > 60000:  # Adjust this threshold as needed
                inf_region_mask = inf_region_mask | region_mask
            elif np.min(new_depth) < 1000:  # Adjust this threshold as needed
                close_region_mask = close_region_mask | region_mask
            else:
                new_depth_image[region_mask > 0] = new_depth.astype(np.uint16)
                complete_region_mask = complete_region_mask | region_mask

            # Visualization (optional)
            if visualize:
                combined[:, :, 0] = rgb_image[:, :, 0] / 2 + thin_edges / 4
                combined[:, :, 1] = rgb_image[:, :, 1] / 2 + complete_region_mask / 2
                combined[:, :, 2] = rgb_image[:, :, 2] / 2 + inf_region_mask / 2
                plt.clf()
                plt.imshow(combined)
                plt.title(f"Processing: {startx}, {starty}")
                plt.draw()
                plt.pause(0.0001)

    new_depth_image[:250] = 0
    new_depth_image[-250:] = 0

    if visualize:
        plt.close()

    return new_depth_image, complete_region_mask, edges_mask, inf_region_mask, close_region_mask

def find_region(startx, starty, rgb_image, step, thin_mask):
    h, w = rgb_image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    
    flood_mask = np.zeros((h+2, w+2), np.uint8)
    flood_mask[1:-1, 1:-1] = thin_mask
    
    cv2.floodFill(rgb_image, flood_mask, (startx, starty), 
                  newVal=(255, 0, 0), 
                  loDiff=(5, 5, 5), upDiff=(5, 5, 5), 
                  flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
    
    mask = flood_mask[1:-1, 1:-1]
    mask = mask & ~thin_mask
    
    return mask

def find_limits_mono(depth_image):
    valid_depths = depth_image[depth_image > 0]
    
    if len(valid_depths) == 0:
        return 0, 0, 0, 0, 0, 0

    front_dist = np.percentile(valid_depths, 1)
    back_dist = np.percentile(valid_depths, 99)

    h, w = depth_image.shape
    vertical_slice = depth_image[:, w//2]
    valid_vertical = vertical_slice[vertical_slice > 0]
    ceil_height = np.percentile(valid_vertical, 1)
    floor_height = np.percentile(valid_vertical, 99)

    horizontal_slice = depth_image[h//2, :]
    valid_horizontal = horizontal_slice[horizontal_slice > 0]
    left_dist = np.percentile(valid_horizontal, 1)
    right_dist = np.percentile(valid_horizontal, 99)

    scale_factor = 0.0002
    ceil_height *= scale_factor
    floor_height *= scale_factor
    front_dist *= scale_factor
    back_dist *= scale_factor
    left_dist *= scale_factor
    right_dist *= scale_factor

    return ceil_height, floor_height, front_dist, back_dist, left_dist, right_dist

def fix_limits_mono(depth_image, ceil_height, floor_height, front_dist, back_dist, left_dist, right_dist):
    scale_factor = 0.0002
    front_dist_scaled = front_dist / scale_factor
    back_dist_scaled = back_dist / scale_factor

    depth_image = np.clip(depth_image, front_dist_scaled, back_dist_scaled)

    depth_range = back_dist_scaled - front_dist_scaled
    normalized_depth = (depth_image - front_dist_scaled) / depth_range
    depth_image = (normalized_depth * 65535).astype(np.uint16)

    return depth_image

def apply_gradient_fade(depth_image, fade_pixels=250):
    h, w = depth_image.shape
    
    top_gradient = np.linspace(0, 1, fade_pixels)[:, np.newaxis]
    bottom_gradient = np.linspace(1, 0, fade_pixels)[:, np.newaxis]
    
    depth_image[:fade_pixels] = depth_image[:fade_pixels] * top_gradient + 65535 * (1 - top_gradient)
    depth_image[-fade_pixels:] = depth_image[-fade_pixels:] * bottom_gradient + 65535 * (1 - bottom_gradient)
    
    return depth_image

def process(input_dir, depth_file, rgb_file, out_depth_file, visualize=False):
    depth_path = os.path.join(input_dir, depth_file)
    rgb_path = os.path.join(input_dir, rgb_file)
    out_path = os.path.join(input_dir, out_depth_file)

    print(f"Processing depth file: {depth_path}")
    print(f"Processing RGB file: {rgb_path}")
    print(f"Output will be saved to: {out_path}")

    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        print(f"Error: Could not read depth image from {depth_path}")
        return

    rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_image is None:
        print(f"Error: Could not read RGB image from {rgb_path}")
        return

    print(f"Depth image shape: {depth_image.shape}, dtype: {depth_image.dtype}")
    print(f"RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")

    if len(depth_image.shape) > 2:
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        print(f"Converted depth image to single channel. New shape: {depth_image.shape}")

    if depth_image.dtype != np.uint16:
        depth_image = (depth_image.astype(np.float32) * 65535 / 255).astype(np.uint16)
        print(f"Converted depth image to uint16. New dtype: {depth_image.dtype}")

    # Edge detection
    edges_image = cv2.Canny(rgb_image, 100, 200)
    kernel = np.ones((5,5), np.uint8)
    thin_edges = cv2.dilate(edges_image, kernel, iterations=1)

    # Apply improved find_planes_mono
    enhanced_depth, complete_region_mask, edges_mask, inf_region_mask, close_region_mask = find_planes_mono(
        depth_image, rgb_image, edges_image, thin_edges, visualize=visualize
    )

    ceil_height, floor_height, front_dist, back_dist, left_dist, right_dist = find_limits_mono(enhanced_depth)

    print(f"Estimated room dimensions:")
    print(f"Height: {floor_height - ceil_height:.2f} meters")
    print(f"Width: {right_dist - left_dist:.2f} meters")
    print(f"Length: {back_dist - front_dist:.2f} meters")

    fixed_depth = fix_limits_mono(
        enhanced_depth, 
        ceil_height, floor_height, front_dist, back_dist, left_dist, right_dist
    )

    fixed_depth = apply_gradient_fade(fixed_depth, fade_pixels=250)

    cv2.imwrite(out_path, fixed_depth)
    print(f"Saved enhanced depth map to: {out_path}")

    fixed_depth_visible = cv2.normalize(fixed_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    visible_out_path = out_path.replace('.png', '_visible.png')
    cv2.imwrite(visible_out_path, fixed_depth_visible)
    print(f"Saved visible enhanced depth map to: {visible_out_path}")


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="360 mono depth map enhancer")
    parser.add_argument("depth_map", help="Input depth map filename", type=str)
    parser.add_argument("rgb", help="Input RGB image filename", type=str)
    parser.add_argument("output", help="Output depth map filename", type=str)
    parser.add_argument("--visualize", help="Enable visualization", action="store_true")
    return parser.parse_args()

def main():
    args = parse_arguments()
    input_dir = os.path.join("Data", "Input")
    process(input_dir, args.depth_map, args.rgb, args.output, visualize=args.visualize)

if __name__ == '__main__':
    main()