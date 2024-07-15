import argparse
import cv2
import os
import numpy as np

# Define the data path
DATA_PATH = './Data'

def apply_gradient_to_white(image, gradient_height):
    h, w = image.shape[:2]
    
    # Create top gradient
    top_gradient = np.linspace(0, 1, gradient_height)[:, np.newaxis]
    top_gradient = np.tile(top_gradient, (1, w))
    
    # Create bottom gradient
    bottom_gradient = np.linspace(1, 0, gradient_height)[:, np.newaxis]
    bottom_gradient = np.tile(bottom_gradient, (1, w))
    
    # Apply gradient (fade to white)
    image[:gradient_height] = image[:gradient_height] * top_gradient + 255 * (1 - top_gradient)
    image[-gradient_height:] = image[-gradient_height:] * bottom_gradient + 255 * (1 - bottom_gradient)
    
    return image

def process(depth_file, out_depth_file):
    print(f"Attempting to read depth image from: {depth_file}")
    
    if not os.path.exists(depth_file):
        raise FileNotFoundError(f"The file {depth_file} does not exist.")
    
    # Read the depth image
    depth_image = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    
    if depth_image is None:
        raise ValueError(f"Failed to read the image file: {depth_file}")
    
    print(f"Depth image shape: {depth_image.shape}")
    print(f"Depth image type: {depth_image.dtype}")
    print(f"Depth image min value: {np.min(depth_image)}")
    print(f"Depth image max value: {np.max(depth_image)}")

    # If the image has multiple channels, use only the first channel
    if len(depth_image.shape) > 2:
        depth_image = depth_image[:,:,0]
    
    # Convert to float32 for processing
    depth_image = depth_image.astype(np.float32)

    # Create a copy of the depth image
    new_depth_image = depth_image.copy()

    # Apply gradient to top and bottom regions
    gradient_height = 250
    new_depth_image = apply_gradient_to_white(new_depth_image, gradient_height)

    # Convert back to uint8
    new_depth_image = np.clip(new_depth_image, 0, 255).astype(np.uint8)

    print(f"Processed image shape: {new_depth_image.shape}")
    print(f"Processed image type: {new_depth_image.dtype}")
    print(f"Processed image min value: {np.min(new_depth_image)}")
    print(f"Processed image max value: {np.max(new_depth_image)}")

    print(f"Saving processed depth image to: {out_depth_file}")
    
    # Save the processed depth image
    cv2.imwrite(out_depth_file, new_depth_image)
    
    if not os.path.exists(out_depth_file):
        raise IOError(f"Failed to save the processed image to {out_depth_file}")
    
    print("Processing completed successfully.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="360 mono depth map enhancer")
    parser.add_argument("input_dir", help="Input directory", type=str)
    parser.add_argument("depth_map", help="Input depth map filename", type=str)
    parser.add_argument("output", help="Output depth map filename", type=str)
    parser.add_argument("--data_path", help=f"Data path. Default {DATA_PATH}", type=str,
                        default=DATA_PATH, required=False)
    
    args = parser.parse_args()
    
    # Use the provided or default DATA_PATH
    data_path = args.data_path
    
    depth_file = os.path.join(data_path, args.input_dir, args.depth_map)
    out_depth_file = os.path.join(data_path, args.input_dir, args.output)
    
    print(f"Input depth file path: {depth_file}")
    print(f"Output depth file path: {out_depth_file}")
    
    return depth_file, out_depth_file

def main():
    try:
        depth_file, out_depth_file = parse_arguments()
        process(depth_file, out_depth_file)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print("Directory contents:")
        for root, dirs, files in os.walk("."):
            for file in files:
                print(os.path.join(root, file))

if __name__ == '__main__':
    main()