import os
import cv2
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from transform_utils import random_paper_augmentation_for_unet


def process_single_id(
    id_num,
    train_dir,
    mask_dir,
    output_images_dir,
    output_masks_dir,
    output_point_masks_dir,
    grid_points,
    num_augmentations,
    pad_ratio,
    max_rotation_deg
):
    """Process a single ID and generate augmentations."""
    img_path = os.path.join(train_dir, str(id_num), f"{id_num}-0001.png")
    mask_path = os.path.join(mask_dir, str(id_num), 'mask', f"{id_num}_mask.png")
    
    if not os.path.exists(img_path):
        return (id_num, 0, f"Image not found: {img_path}")
    
    if not os.path.exists(mask_path):
        return (id_num, 0, f"Mask not found: {mask_path}")
    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    
    if img is None:
        return (id_num, 0, "Failed to load image")
    
    if mask is None:
        return (id_num, 0, "Failed to load mask")
    
    if img.shape[:2] != mask.shape[:2]:
        return (id_num, 0, f"Dimension mismatch - Image: {img.shape}, Mask: {mask.shape}")
    
    generated = 0
    for aug_idx in range(num_augmentations):
        try:
            aug_img, aug_mask, aug_grid_mask = random_paper_augmentation_for_unet(
                img,
                mask,
                grid_points,
                pad_ratio=pad_ratio,
                max_rotation_deg=max_rotation_deg
            )
            
            aug_img_filename = f"{id_num}_aug_{aug_idx:02d}.png"
            aug_mask_filename = f"{id_num}_aug_{aug_idx:02d}_mask.png"
            aug_grid_mask_filename = f"{id_num}_aug_{aug_idx:02d}_grid_point_mask.png"
            
            aug_img_path = os.path.join(output_images_dir, aug_img_filename)
            aug_mask_path = os.path.join(output_masks_dir, aug_mask_filename)
            aug_grid_mask_path = os.path.join(output_point_masks_dir, aug_grid_mask_filename)
            
            cv2.imwrite(aug_img_path, aug_img)
            cv2.imwrite(aug_mask_path, aug_mask)
            cv2.imwrite(aug_grid_mask_path, aug_grid_mask)
            
            generated += 1
            
        except Exception as e:
            return (id_num, generated, f"Error in augmentation {aug_idx}: {str(e)}")
    
    return (id_num, generated, None)


def generate_augmented_dataset(
    train_dir,
    mask_dir,
    output_dir,
    grid_csv_path,
    num_augmentations=1,
    pad_ratio=0.08,
    max_rotation_deg=1.0,
    num_workers=13
):
    """Generate augmented dataset using paper simulation transforms with parallel processing."""
    
    output_images_dir = os.path.join(output_dir, 'images')
    output_masks_dir = os.path.join(output_dir, 'masks')
    output_point_masks_dir = os.path.join(output_dir, 'point_masks')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    os.makedirs(output_point_masks_dir, exist_ok=True)
    
    if not os.path.exists(train_dir):
        print(f"Error: Train directory not found: {train_dir}")
        return
    
    ids = []
    for item in os.listdir(train_dir):
        item_path = os.path.join(train_dir, item)
        if os.path.isdir(item_path):
            try:
                ids.append(int(item))
            except ValueError:
                print(f"Skipping non-numeric folder: {item}")
    
    ids.sort()
    
    if not os.path.exists(grid_csv_path):
        print(f"Error: Grid CSV not found: {grid_csv_path}")
        return
    
    print(f"Loading grid coordinates from {grid_csv_path}...")
    try:
        grid_df = pd.read_csv(grid_csv_path)
        if grid_df.isnull().values.any():
            print("Warning: Grid CSV contains NaNs. Filling with 0.")
            grid_df = grid_df.fillna(0)
            
        grid_points = grid_df[['x', 'y']].values.astype(np.float32)
        print(f"Loaded {len(grid_points)} grid points.")
        print(f"Coordinate Range - X: [{grid_points[:,0].min()}, {grid_points[:,0].max()}], Y: [{grid_points[:,1].min()}, {grid_points[:,1].max()}]")
        
    except Exception as e:
        print(f"Error loading grid CSV: {e}")
        return

    print(f"Found {len(ids)} IDs to process")
    print(f"Using {num_workers} parallel workers")
    
    process_func = partial(
        process_single_id,
        train_dir=train_dir,
        mask_dir=mask_dir,
        output_images_dir=output_images_dir,
        output_masks_dir=output_masks_dir,
        output_point_masks_dir=output_point_masks_dir,
        grid_points=grid_points,
        num_augmentations=num_augmentations,
        pad_ratio=pad_ratio,
        max_rotation_deg=max_rotation_deg
    )
    
    total_generated = 0
    skipped = 0
    errors = []
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, ids),
            total=len(ids),
            desc="Processing IDs"
        ))
    
    for id_num, num_generated, error_msg in results:
        if error_msg:
            print(f"Warning: ID {id_num} - {error_msg}")
            errors.append((id_num, error_msg))
            skipped += 1
        else:
            total_generated += num_generated
    
    print(f"\nDataset generation complete")
    print(f"Total IDs processed: {len(ids)}")
    print(f"Total augmented pairs generated: {total_generated}")
    print(f"Successfully processed: {len(ids) - skipped}")
    print(f"Skipped: {skipped}")
    if errors:
        print(f"\nErrors encountered: {len(errors)}")
        for id_num, error_msg in errors[:10]:
            print(f"  ID {id_num}: {error_msg}")
    print(f"\nOutput saved to:")
    print(f"  Images: {output_images_dir}")
    print(f"  Masks: {output_masks_dir}")
    print(f"  Point Masks: {output_point_masks_dir}")


def main():
    base_dir = '/Users/felix/Documents/ecg_12_1'
    train_dir = os.path.join(base_dir, 'train')
    mask_dir = os.path.join(base_dir, 'gen_train')
    grid_csv_path = os.path.join(base_dir, 'grid_coordinates.csv')
    output_dir = os.path.join(base_dir, 'generated_dataset')
    
    num_augmentations = 1
    pad_ratio = 0.08
    max_rotation_deg = 1.0
    num_workers = 13
    
    print("Starting dataset augmentation with parallel processing...")
    print(f"Train directory: {train_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Grid CSV: {grid_csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Augmentations per image: {num_augmentations}")
    print(f"Parallel workers: {num_workers}")
    print()
    
    generate_augmented_dataset(
        train_dir=train_dir,
        mask_dir=mask_dir,
        output_dir=output_dir,
        grid_csv_path=grid_csv_path,
        num_augmentations=num_augmentations,
        pad_ratio=pad_ratio,
        max_rotation_deg=max_rotation_deg,
        num_workers=num_workers
    )


if __name__ == "__main__":
    main()
