import os
import torch
import sys
# Add the parent directory of `mast3r` to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "mast3r")))
from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from dust3r.utils.image import load_images
from mast3r.fast_nn import fast_reciprocal_NNs

import os
import torch
import sys

# Add the parent directory of `mast3r` to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "mast3r")))
from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from dust3r.utils.image import load_images
from mast3r.fast_nn import fast_reciprocal_NNs

def run_mast3r(
    image_folder="/my_vol/mast3r_sfm/data/images",
    model_checkpoint="/my_vol/mast3r_sfm/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    output_dir="/my_vol/mast3r_sfm/output/",
    image_size=(512, 512),
    device="cuda"
):
    """
    Run MASt3R on a folder of images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure image_size is a tuple
    if not isinstance(image_size, tuple):
        raise ValueError(f"Expected `image_size` as a tuple, got {type(image_size)}")

    # Load the model
    if os.path.isfile(model_checkpoint):
        print(f"Loading model from local checkpoint: {model_checkpoint}")
        model = AsymmetricMASt3R.from_pretrained(model_checkpoint).to(device)
    else:
        raise ValueError(f"Model checkpoint not found at {model_checkpoint}")

    # Load the images
    print(f"Loading images from {image_folder}...")
    image_files = sorted(
        [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".png", ".jpg"))]
    )
    images = load_images(image_files, size=max(image_size))  # Pass max dimension for resizing

    # Create pairs for stereo inference
    pairs = [(images[i], images[i + 1]) for i in range(len(images) - 1)]
    print(f"Number of pairs created: {len(pairs)}")

    # Run inference
    print("Running inference...")
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    # Process results (you can modify this as needed)
    print("Processing results...")
    # Example: Save output directly to output_dir
    output_file = os.path.join(output_dir, "output.pt")
    torch.save(output, output_file)
    print(f"Results saved to {output_file}")

    return output_file



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MASt3R on images.")
    parser.add_argument("--image_folder", type=str, default="data/images", help="Path to the folder of images.")
    parser.add_argument("--model_checkpoint", type=str, default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth", help="Path to the pretrained MASt3R checkpoint.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs.")
    parser.add_argument("--image_size", type=str, default="512,512", help="Image size for resizing, specified as 'height,width'.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")
    args = parser.parse_args()

    # Convert `image_size` from string to a tuple of integers
    image_size = tuple(map(int, args.image_size.split(",")))

    # Run the function
    run_mast3r(
        image_folder=args.image_folder,
        model_checkpoint=args.model_checkpoint,
        output_dir=args.output_dir,
        image_size=image_size,
        device=args.device
    )
