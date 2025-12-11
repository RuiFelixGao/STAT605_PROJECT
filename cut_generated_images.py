#!/usr/bin/env python3
"""
Batch Document Scanner with Multi-threading
Process all images in ./generated_dataset/images/image_{00-15} and corresponding masks and point_masks using Canny method
"""

import cv2
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from segment_utils import DocumentScanner


def process_single_image(image_path, mask_dir, point_mask_dir):
    """
    Process a single image and its corresponding mask and point_mask using Canny method.
    The mask and point_mask are transformed using the same corners detected from the image.
    
    Args:
        image_path: Path object pointing to the image file
        mask_dir: Path object pointing to the masks directory
        point_mask_dir: Path object pointing to the point_masks directory
        
    Returns:
        tuple: (image_name, success_status, error_message)
    """
    try:
        scanner = DocumentScanner()
        
        processed_img, corners = scanner.process_image(
            str(image_path), 
            method="canny"
        )
        
        image_name = image_path.stem
        mask_name = f"{image_name}_mask.png"
        point_mask_name = f"{image_name}_grid_point_mask.png"
        mask_path = mask_dir / mask_name
        point_mask_path = point_mask_dir / point_mask_name
        
        if not mask_path.exists():
            return (image_path.name, False, f"Mask not found: {mask_name}")
        
        if not point_mask_path.exists():
            return (image_path.name, False, f"Point mask not found: {point_mask_name}")
        
        mask_img = cv2.imread(str(mask_path))
        if mask_img is None:
            return (image_path.name, False, f"Failed to load mask: {mask_name}")
        
        point_mask_img = cv2.imread(str(point_mask_path))
        if point_mask_img is None:
            return (image_path.name, False, f"Failed to load point mask: {point_mask_name}")
        
        original_img = cv2.imread(str(image_path))
        h, w = original_img.shape[:2]
        default_corners = [[0, 0], [w, 0], [w, h], [0, h]]
        
        corners_are_default = all(
            abs(corners[i][0] - default_corners[i][0]) < 1 and 
            abs(corners[i][1] - default_corners[i][1]) < 1 
            for i in range(4)
        )

        if not corners_are_default:
            ph, pw = processed_img.shape[:2]
            if ph < 1500 or pw < 2000:
                corners_are_default = True

        if corners_are_default:
            processed_mask = mask_img
            processed_point_mask = point_mask_img
            processed_img = original_img
        else:
            processed_mask = scanner.rotate_and_crop(mask_img, corners)
            processed_point_mask = scanner.rotate_and_crop(point_mask_img, corners)

        cv2.imwrite(str(image_path), processed_img)
        cv2.imwrite(str(mask_path), processed_mask)
        cv2.imwrite(str(point_mask_path), processed_point_mask)
        
        return (image_path.name, True, None)
        
    except Exception as e:
        return (image_path.name, False, str(e))


def main():
    NUM_WORKERS = 13
    BASE_IMAGE_DIR = Path("./generated_dataset/images")
    BASE_MASK_DIR = Path("./generated_dataset/masks")
    BASE_POINT_MASK_DIR = Path("./generated_dataset/point_masks")
    
    if not BASE_IMAGE_DIR.exists():
        print(f"Base image directory not found: {BASE_IMAGE_DIR}")
        sys.exit(1)
    
    if not BASE_MASK_DIR.exists():
        print(f"Base mask directory not found: {BASE_MASK_DIR}")
        sys.exit(1)
    
    if not BASE_POINT_MASK_DIR.exists():
        print(f"Base point mask directory not found: {BASE_POINT_MASK_DIR}")
        sys.exit(1)
    
    total_success_count = 0
    total_error_count = 0
    total_errors = []
    
    for folder_num in range(16):
        folder_name = f"{folder_num:02d}"
        IMAGE_DIR = BASE_IMAGE_DIR / f"image_{folder_name}"
        MASK_DIR = BASE_MASK_DIR / f"mask_{folder_name}"
        POINT_MASK_DIR = BASE_POINT_MASK_DIR / f"point_mask_{folder_name}"
        
        print(f"\nProcessing folder {folder_num + 1}/16: image_{folder_name}")
        
        if not IMAGE_DIR.exists():
            print(f"Image directory not found: {IMAGE_DIR}")
            print(f"Skipping folder {folder_name}...")
            continue
        
        if not MASK_DIR.exists():
            print(f"Mask directory not found: {MASK_DIR}")
            print(f"Skipping folder {folder_name}...")
            continue
        
        if not POINT_MASK_DIR.exists():
            print(f"Point mask directory not found: {POINT_MASK_DIR}")
            print(f"Skipping folder {folder_name}...")
            continue
        
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(IMAGE_DIR.glob(ext))
        
        if not image_files:
            print(f"No image files found in {IMAGE_DIR}")
            print(f"Skipping folder {folder_name}...")
            continue
        
        print(f"Found {len(image_files)} images to process")
        print(f"Using {NUM_WORKERS} worker threads")
        print(f"Method: Canny edge detection")
        print(f"Processing: Images, masks, and point_masks")
        print(f"Output: Overwriting original files")
        
        success_count = 0
        error_count = 0
        errors = []
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(process_single_image, img, MASK_DIR, POINT_MASK_DIR): img 
                       for img in image_files}
            
            with tqdm(total=len(image_files), desc=f"Folder {folder_name}", unit="img") as pbar:
                for future in as_completed(futures):
                    filename, success, error_msg = future.result()
                    
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        errors.append((folder_name, filename, error_msg))
                    
                    pbar.update(1)
        
        print(f"Folder {folder_name} complete!")
        print(f"  Success: {success_count}/{len(image_files)}")
        print(f"  Failed:  {error_count}/{len(image_files)}")
        
        if errors:
            print(f"\nErrors in folder {folder_name}:")
            for _, filename, error_msg in errors[:5]:
                print(f"  {filename}: {error_msg}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
        
        total_success_count += success_count
        total_error_count += error_count
        total_errors.extend(errors)
    
    print(f"\nAll folders processing complete")
    print(f"Total Success: {total_success_count}")
    print(f"Total Failed:  {total_error_count}")
    
    if total_errors:
        print(f"\nTotal errors encountered: {len(total_errors)}")
        print("\nFirst 10 errors:")
        for folder, filename, error_msg in total_errors[:10]:
            print(f"  {folder}/{filename}: {error_msg}")
        if len(total_errors) > 10:
            print(f"  ... and {len(total_errors) - 10} more errors")


if __name__ == "__main__":
    main()
