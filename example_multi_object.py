"""
SAM 3D Objects - Multi-Object Reconstruction Example

This example demonstrates how to use SAM 3D Objects to reconstruct
multiple objects from a single image and merge them into a unified scene.

Author: Analysis for superlearnerG/sam-3d-objects
Date: December 2025
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add notebook to path for inference utilities
sys.path.append("notebook")
from inference import (
    Inference,
    load_image,
    load_masks,
    load_single_mask,
    display_image,
    make_scene,
    ready_gaussian_for_video_rendering,
    render_video,
)


def reconstruct_single_object(
    inference_model,
    image_path: str,
    mask_index: int,
    output_path: str = "output_single.ply",
    seed: int = 42
):
    """
    Reconstruct a single object from an image.
    
    Args:
        inference_model: Loaded Inference model instance
        image_path: Path to the input image (PNG format)
        mask_index: Index of the mask file to use
        output_path: Path to save the output PLY file
        seed: Random seed for reproducibility
    
    Returns:
        dict: Output dictionary containing reconstruction results
    """
    print(f"Loading image from: {image_path}")
    image = load_image(image_path)
    
    mask_dir = os.path.dirname(image_path)
    print(f"Loading mask {mask_index} from: {mask_dir}")
    mask = load_single_mask(mask_dir, index=mask_index)
    
    print("Running inference...")
    output = inference_model(image, mask, seed=seed)
    
    print(f"Saving reconstruction to: {output_path}")
    output["gs"].save_ply(output_path)
    
    print(f"✓ Single object reconstruction complete!")
    return output


def reconstruct_multiple_objects(
    inference_model,
    image_path: str,
    mask_indices: list = None,
    output_path: str = "output_multi.ply",
    seed: int = 42
):
    """
    Reconstruct multiple objects from a single image and merge them into a scene.
    
    Args:
        inference_model: Loaded Inference model instance
        image_path: Path to the input image (PNG format)
        mask_indices: List of mask indices to use. If None, uses all available masks
        output_path: Path to save the output PLY file
        seed: Random seed for reproducibility
    
    Returns:
        Merged scene with all objects
    """
    print(f"Loading image from: {image_path}")
    image = load_image(image_path)
    
    mask_dir = os.path.dirname(image_path)
    print(f"Loading masks from: {mask_dir}")
    masks = load_masks(mask_dir, indices_list=mask_indices)
    
    print(f"Found {len(masks)} masks to process")
    
    # Display image with masks for verification
    display_image(image, masks)
    
    print("Running inference for each object...")
    outputs = []
    for i, mask in enumerate(masks):
        print(f"  Processing object {i+1}/{len(masks)}...")
        output = inference_model(image, mask, seed=seed)
        outputs.append(output)
    
    print("Merging objects into unified scene...")
    scene_gs = make_scene(*outputs)
    scene_gs = ready_gaussian_for_video_rendering(scene_gs)
    
    print(f"Saving scene to: {output_path}")
    scene_gs.save_ply(output_path)
    
    print(f"✓ Multi-object reconstruction complete!")
    print(f"  Total objects: {len(outputs)}")
    return scene_gs


def main():
    """
    Main function demonstrating single and multi-object reconstruction.
    """
    # Configuration
    TAG = "hf"  # Model tag (download from HuggingFace)
    config_path = f"checkpoints/{TAG}/pipeline.yaml"
    
    # Example paths (modify these to your actual data)
    example_image = "notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png"
    
    print("=" * 60)
    print("SAM 3D Objects - Multi-Object Reconstruction Example")
    print("=" * 60)
    
    # Load the inference model
    print("\n[1/3] Loading SAM 3D Objects model...")
    print(f"Config: {config_path}")
    
    if not os.path.exists(config_path):
        print("\n❌ Error: Model checkpoint not found!")
        print("Please download the model first:")
        print("  1. Request access at: https://huggingface.co/facebook/sam-3d-objects")
        print("  2. Run: hf download --repo-type model --local-dir checkpoints/hf-download facebook/sam-3d-objects")
        print("  3. Run: mv checkpoints/hf-download/checkpoints checkpoints/hf")
        return
    
    inference = Inference(config_path, compile=False)
    print("✓ Model loaded successfully")
    
    # Example 1: Single object reconstruction
    print("\n[2/3] Example 1: Single Object Reconstruction")
    print("-" * 60)
    if os.path.exists(example_image):
        single_output = reconstruct_single_object(
            inference_model=inference,
            image_path=example_image,
            mask_index=14,  # Specific object in the scene
            output_path="output_single_object.ply",
            seed=42
        )
    else:
        print(f"⚠ Example image not found: {example_image}")
        print("Please provide your own image and masks.")
    
    # Example 2: Multi-object reconstruction
    print("\n[3/3] Example 2: Multi-Object Reconstruction")
    print("-" * 60)
    if os.path.exists(example_image):
        # Reconstruct all objects in the scene
        # Note: This will process ALL mask files (0.png, 1.png, 2.png, ...)
        # found in the image directory
        scene = reconstruct_multiple_objects(
            inference_model=inference,
            image_path=example_image,
            mask_indices=None,  # Use all available masks
            output_path="output_multi_object_scene.ply",
            seed=42
        )
        
        print("\n" + "=" * 60)
        print("Reconstruction complete! Output files:")
        print("  - output_single_object.ply (single object)")
        print("  - output_multi_object_scene.ply (full scene)")
        print("\nYou can visualize these files using:")
        print("  - CloudCompare: https://www.danielgm.net/cc/")
        print("  - MeshLab: https://www.meshlab.net/")
        print("  - Online viewers: https://playcanvas.com/viewer")
    else:
        print(f"⚠ Example image not found: {example_image}")
        print("Please provide your own image and masks.")
    
    print("\n" + "=" * 60)
    print("For more examples, check:")
    print("  - notebook/demo_single_object.ipynb")
    print("  - notebook/demo_multi_object.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    main()
