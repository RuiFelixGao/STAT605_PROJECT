#!/usr/bin/env python3
"""
ECG Segmentation Inference Script with Document Scanning
Uses document scanning and sliding window for inference
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from tqdm import tqdm
from segment_utils import DocumentScanner


class SlidingWindowInference:
    """Sliding window inference with weighted fusion for overlapping regions."""
    
    def __init__(self, model, window_size=512, overlap=128, device='cuda'):
        """
        Args:
            model: Segmentation model
            window_size: Window size (square)
            overlap: Overlap size
            device: Inference device
        """
        self.model = model
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap
        self.device = device
        
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        self.weight_map = self._create_weight_map()
        
    def _create_weight_map(self):
        """Create Gaussian weight map for smooth fusion of overlapping regions."""
        center = self.window_size // 2
        y, x = np.ogrid[:self.window_size, :self.window_size]
        
        sigma = self.window_size / 4
        weight = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
        
        weight = (weight - weight.min()) / (weight.max() - weight.min())
        weight = weight * 0.9 + 0.1
        
        return weight.astype(np.float32)
    
    def _preprocess(self, image):
        """Preprocess image."""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        return image
    
    def _postprocess(self, mask):
        """Postprocess mask."""
        mask = (mask * 255).astype(np.uint8)
        return mask
    
    def predict(self, image, use_amp=True):
        """
        Perform sliding window inference on image.
        
        Args:
            image: numpy array (H, W, 3) BGR format
            use_amp: Use mixed precision
            
        Returns:
            mask: Segmentation result (H, W, 3)
        """
        h, w = image.shape[:2]
        print(f"Inference on image size: {w}x{h}")
        
        image_norm = self._preprocess(image)
        
        output = np.zeros((h, w, 3), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)
        
        n_rows = (h - self.overlap) // self.stride + 1
        n_cols = (w - self.overlap) // self.stride + 1
        total_windows = n_rows * n_cols
        
        print(f"Sliding window: {self.window_size}x{self.window_size}, stride: {self.stride}")
        print(f"Total windows: {total_windows} ({n_rows} rows x {n_cols} cols)")
        
        self.model.eval()
        
        with torch.no_grad():
            pbar = tqdm(total=total_windows, desc="Inference")
            
            for i in range(n_rows):
                for j in range(n_cols):
                    y1 = i * self.stride
                    x1 = j * self.stride
                    y2 = min(y1 + self.window_size, h)
                    x2 = min(x1 + self.window_size, w)
                    
                    if y2 - y1 < self.window_size:
                        y1 = max(0, y2 - self.window_size)
                    if x2 - x1 < self.window_size:
                        x1 = max(0, x2 - self.window_size)
                    
                    window = image_norm[y1:y2, x1:x2]
                    
                    window_tensor = torch.from_numpy(window.transpose(2, 0, 1)).unsqueeze(0)
                    window_tensor = window_tensor.to(self.device)
                    
                    if use_amp and torch.cuda.is_available():
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            pred = self.model(window_tensor)
                    else:
                        pred = self.model(window_tensor)
                    
                    pred = torch.sigmoid(pred)
                    pred = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    
                    weight = self.weight_map[:pred.shape[0], :pred.shape[1]]
                    output[y1:y2, x1:x2] += pred * weight[:, :, np.newaxis]
                    weight_sum[y1:y2, x1:x2] += weight
                    
                    pbar.update(1)
            
            pbar.close()
        
        weight_sum = np.maximum(weight_sum, 1e-8)
        output = output / weight_sum[:, :, np.newaxis]
        
        mask = self._postprocess(output)
        
        return mask


def resize_if_needed(image, max_width=2500, max_height=2000):
    """
    Resize image if it exceeds thresholds.
    
    Args:
        image: Input image
        max_width: Maximum width threshold
        max_height: Maximum height threshold
        
    Returns:
        resized_image, scale_factor
    """
    h, w = image.shape[:2]
    
    if h > max_height or w > 2600:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        
        print(f"Resizing image from {w}x{h} to {new_w}x{new_h} (scale: {scale:.3f})")
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, scale
    else:
        print(f"Image size {w}x{h} is within limits, no resize needed")
        return image, 1.0


def crop_top_fifth(image):
    """
    Crop top 1/5 of image, keep bottom 4/5.
    
    Args:
        image: Input image
        
    Returns:
        cropped_image
    """
    h, w = image.shape[:2]
    crop_start = int(h * 0.2)
    
    print(f"Cropping top 1/5: from {w}x{h} to {w}x{h - crop_start}")
    print(f"Removing top {crop_start} pixels, keeping bottom {h - crop_start} pixels")
    
    cropped = image[crop_start:, :].copy()
    return cropped


def main():
    parser = argparse.ArgumentParser(description='ECG Segmentation Inference with Document Scanning')
    parser.add_argument('image_path', type=str, nargs='?', default='samples/sample1.png',
                        help='Path to input image (default: samples/sample1.png)')
    parser.add_argument('--checkpoint', type=str, default='unet_model_1209.pth',
                        help='Path to model checkpoint (default: unet_model_1209.pth)')
    parser.add_argument('--window-size', type=int, default=512,
                        help='Sliding window size (default: 512)')
    parser.add_argument('--overlap', type=int, default=128,
                        help='Overlap between windows (default: 128)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: cuda/mps/cpu/auto (default: auto)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binary threshold for visualization (default: 0.5)')
    parser.add_argument('--no-segment', action='store_true',
                        help='Skip document segmentation step')
    parser.add_argument('--segment-method', type=str, default='canny',
                        choices=['canny', 'dark_border'],
                        help='Document segmentation method (default: canny)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(args.device)
        print(f"Using device: {device}")
    
    input_dir = os.path.dirname(os.path.abspath(args.image_path))
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    
    print(f"\nStep 1: Document Segmentation")
    
    original_image = cv2.imread(args.image_path)
    if original_image is None:
        print(f"Error: Failed to load image: {args.image_path}")
        return
    
    print(f"Original image size: {original_image.shape[1]}x{original_image.shape[0]}")
    
    if not args.no_segment:
        scanner = DocumentScanner()
        try:
            cropped_image, corner_pts = scanner.process_image(args.image_path, method=args.segment_method)
            print(f"Document segmentation completed")
            print(f"Cropped image size: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
            
            cropped_path = os.path.join(input_dir, f"{base_name}_cropped.png")
            cv2.imwrite(cropped_path, cropped_image)
            print(f"Cropped image saved to: {cropped_path}")
            
        except Exception as e:
            print(f"Document segmentation failed: {e}")
            print(f"Using original image")
            cropped_image = original_image
    else:
        print("Document segmentation skipped")
        cropped_image = original_image
    
    print(f"\nStep 2: Image Resizing")
    
    resized_image, scale_factor = resize_if_needed(cropped_image, max_width=2500, max_height=2000)
    
    print(f"\nStep 2.5: Crop Top 1/5")
    
    final_image = crop_top_fifth(resized_image)
    
    print(f"\nStep 3: Model Loading")
    
    print(f"Loading model from {args.checkpoint}...")
    model = smp.Unet("resnet34", in_channels=3, classes=3, encoder_weights=None)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights")
    
    model = model.to(device)
    model.eval()
    
    print(f"\nStep 4: Sliding Window Inference")
    
    inferencer = SlidingWindowInference(
        model=model,
        window_size=args.window_size,
        overlap=args.overlap,
        device=device
    )
    
    mask = inferencer.predict(final_image, use_amp=not args.no_amp)
    
    print(f"\nStep 5: Saving Results")
    
    output_path = os.path.join(input_dir, f"{base_name}_mask.png")
    
    cv2.imwrite(output_path, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
    print(f"Mask saved to: {output_path}")
    
    binary_mask = (mask > args.threshold * 255).astype(np.uint8) * 255
    binary_output = os.path.join(input_dir, f"{base_name}_mask_binary.png")
    cv2.imwrite(binary_output, cv2.cvtColor(binary_mask, cv2.COLOR_RGB2BGR))
    print(f"Binary mask saved to: {binary_output}")
    
    overlay = final_image.copy()
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(final_image, 0.6, mask_bgr, 0.4, 0)
    
    overlay_output = os.path.join(input_dir, f"{base_name}_mask_overlay.png")
    cv2.imwrite(overlay_output, overlay)
    print(f"Overlay visualization saved to: {overlay_output}")
    
    processed_image_path = os.path.join(input_dir, f"{base_name}_processed.png")
    cv2.imwrite(processed_image_path, final_image)
    print(f"Processed image saved to: {processed_image_path}")
    
    comparison = np.hstack([final_image, overlay])
    comparison_path = os.path.join(input_dir, f"{base_name}_comparison.png")
    cv2.imwrite(comparison_path, comparison)
    print(f"Comparison image saved to: {comparison_path}")
    
    print(f"\nProcessing Summary")
    print(f"Original size:  {original_image.shape[1]}x{original_image.shape[0]}")
    print(f"Cropped size:   {cropped_image.shape[1]}x{cropped_image.shape[0]}")
    print(f"Resized size:   {resized_image.shape[1]}x{resized_image.shape[0]}")
    print(f"Final size:     {final_image.shape[1]}x{final_image.shape[0]}")
    print(f"Scale factor:   {scale_factor:.3f}")
    print(f"\nOutput files:")
    if not args.no_segment:
        print(f"  {cropped_path}")
    print(f"  {processed_image_path}")
    print(f"  {output_path}")
    print(f"  {binary_output}")
    print(f"  {overlay_output}")
    print(f"  {comparison_path}")
    print(f"\nInference completed successfully")


if __name__ == "__main__":
    main()
