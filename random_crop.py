"""
Preprocessing script: Random crop for all images.
Crops image_00~15, corresponding masks, and point_masks into 512x512 patches.
Saves to image_sep_{number}, mask_sep_{number}, and point_mask_sep_{number} folders.
"""

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

CROP_SIZE = 512
CROPS_PER_IMAGE = 20
NUM_WORKERS = cpu_count() - 3


def random_crop_with_seed(img, mask, point_mask, crop_idx, seed_offset):
    """
    Random crop with fixed seed for reproducibility.
    Crops only from the bottom 4/5 of the image.
    """
    h, w = img.shape[:2]
    crop_h, crop_w = CROP_SIZE, CROP_SIZE
    
    np.random.seed(seed_offset + crop_idx)
    
    if h < crop_h or w < crop_w:
        pad_h = max(0, crop_h - h)
        pad_w = max(0, crop_w - w)
        
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        point_mask = cv2.copyMakeBorder(point_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        
        h, w = img.shape[:2]
    
    valid_h_start = h // 5
    
    if h - valid_h_start < crop_h:
        y_min = max(0, h - crop_h)
        y_max = h - crop_h
    else:
        y_min = valid_h_start
        y_max = h - crop_h
    
    if y_max <= y_min:
        y_start = y_min
    else:
        y_start = np.random.randint(y_min, y_max + 1)
        
    if w <= crop_w:
        x_start = 0
    else:
        x_start = np.random.randint(0, w - crop_w + 1)
    
    y1 = y_start
    y2 = y1 + crop_h
    x1 = x_start
    x2 = x1 + crop_w
    
    img_crop = img[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]
    point_mask_crop = point_mask[y1:y2, x1:x2]
    
    np.random.seed(None)
    
    return img_crop, mask_crop, point_mask_crop


def process_single_image(args):
    """Process all crops for a single image and delete original files."""
    img_path, mask_path, point_mask_path, output_img_dir, output_mask_dir, output_point_mask_dir, img_basename = args
    
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        point_mask = cv2.imread(point_mask_path, cv2.IMREAD_GRAYSCALE)
        
        seed_offset = hash(img_path) % 100000
        
        for crop_idx in range(CROPS_PER_IMAGE):
            img_crop, mask_crop, point_mask_crop = random_crop_with_seed(img, mask, point_mask, crop_idx, seed_offset)
            
            crop_basename = img_basename.replace('.png', f'_crop_{crop_idx:02d}.png')
            
            img_crop_bgr = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_img_dir, crop_basename), img_crop_bgr)
            
            mask_crop_bgr = cv2.cvtColor(mask_crop, cv2.COLOR_RGB2BGR)
            mask_crop_basename = crop_basename.replace('.png', '_mask.png')
            cv2.imwrite(os.path.join(output_mask_dir, mask_crop_basename), mask_crop_bgr)
            
            point_mask_crop_basename = crop_basename.replace('.png', '_grid_point_mask.png')
            cv2.imwrite(os.path.join(output_point_mask_dir, point_mask_crop_basename), point_mask_crop)
        
        if os.path.exists(img_path):
            os.remove(img_path)
        if os.path.exists(mask_path):
            os.remove(mask_path)
        if os.path.exists(point_mask_path):
            os.remove(point_mask_path)
            
        return img_basename
        
    except Exception as e:
        print(f"Error processing {img_basename}: {e}")
        return None


def process_number(number_str):
    """Process all images for a single number folder."""
    print(f"\nProcessing number {number_str}")
    
    image_dir = f"./generated_dataset/images/image_{number_str}"
    mask_dir = f"./generated_dataset/masks/mask_{number_str}"
    point_mask_dir = f"./generated_dataset/point_masks/point_mask_{number_str}"
    
    output_img_dir = f"./generated_dataset/images/image_sep_{number_str}"
    output_mask_dir = f"./generated_dataset/masks/mask_sep_{number_str}"
    output_point_mask_dir = f"./generated_dataset/point_masks/point_mask_sep_{number_str}"
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_point_mask_dir, exist_ok=True)
    
    all_imgs = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    
    tasks = []
    for img_path in all_imgs:
        basename = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, basename.replace(".png", "_mask.png"))
        point_mask_path = os.path.join(point_mask_dir, basename.replace(".png", "_grid_point_mask.png"))
        
        if os.path.exists(mask_path) and os.path.exists(point_mask_path):
            tasks.append((img_path, mask_path, point_mask_path, output_img_dir, output_mask_dir, output_point_mask_dir, basename))
        else:
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for {basename}")
            if not os.path.exists(point_mask_path):
                print(f"Warning: Point mask not found for {basename}")
    
    if not tasks:
        print(f"No valid image-mask-point_mask pairs found for number {number_str}")
        return
    
    print(f"Found {len(tasks)} images to process")
    print(f"Each image will generate {CROPS_PER_IMAGE} crops")
    print(f"Total crops to generate: {len(tasks) * CROPS_PER_IMAGE}")
    print(f"Using {NUM_WORKERS} workers")
    
    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, tasks),
            total=len(tasks),
            desc=f"Number {number_str}"
        ))
    
    successful_results = [r for r in results if r is not None]
    
    print(f"Completed number {number_str}")
    print(f"  Output images: {output_img_dir}")
    print(f"  Output masks: {output_mask_dir}")
    print(f"  Output point masks: {output_point_mask_dir}")
    print(f"  Total crops generated: {len(successful_results) * CROPS_PER_IMAGE}")


def main():
    print("ECG Image Preprocessing - Crop Generation")
    print(f"Crop size: {CROP_SIZE}x{CROP_SIZE}")
    print(f"Crops per image: {CROPS_PER_IMAGE}")
    print(f"CPU workers: {NUM_WORKERS}")
    
    for i in range(16):
        number_str = f"{i:02d}"
        
        image_dir = f"./generated_dataset/images/image_{number_str}"
        mask_dir = f"./generated_dataset/masks/mask_{number_str}"
        point_mask_dir = f"./generated_dataset/point_masks/point_mask_{number_str}"
        
        if not os.path.exists(image_dir):
            print(f"\nSkipping {number_str}: {image_dir} not found")
            continue
        
        if not os.path.exists(mask_dir):
            print(f"\nSkipping {number_str}: {mask_dir} not found")
            continue
            
        if not os.path.exists(point_mask_dir):
            print(f"\nSkipping {number_str}: {point_mask_dir} not found")
            continue
        
        process_number(number_str)
    
    print("\nAll preprocessing completed!")
    print("\nNext steps:")
    print("1. Compress each image_sep_{number} folder to tar files")
    print("2. Compress each mask_sep_{number} folder to tar files")
    print("3. Compress each point_mask_sep_{number} folder to tar files")
    print("4. Upload to GPU server and run training script")


if __name__ == "__main__":
    main()
