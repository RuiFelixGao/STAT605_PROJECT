import os
import numpy as np
import pandas as pd
import cv2
import glob
from multiprocessing import Pool, cpu_count

def load_grid_coordinates(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['y', 'x'])
    unique_x = sorted(df['x'].unique())
    unique_y = sorted(df['y'].unique())
    return np.array(unique_x), np.array(unique_y), df

def get_interp_fn(coords):
    indices = np.arange(len(coords))
    return lambda idx: np.interp(idx, indices, coords)

def get_inv_interp_fn(coords):
    indices = np.arange(len(coords))
    return lambda val: np.interp(val, coords, indices)

def load_signal_csv(id, train_dir):
    csv_path = os.path.join(train_dir, str(id), f"{id}.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            return None
    return None

def load_signal(id, sig_len, fs, train_dir):
    npy_path = os.path.join(train_dir, str(id), f"{id}.npy")
    if os.path.exists(npy_path):
        try:
            signal = np.load(npy_path)
            if signal.shape[1] != sig_len and signal.shape[0] == sig_len:
                signal = signal.T
            return signal
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
    
    print(f"Generating placeholder signal for {id}")
    t = np.arange(sig_len) / fs
    signal = np.zeros((12, sig_len))
    for i in range(12):
        freq = 1.0 + np.random.rand() * 2.0
        phase = np.random.rand() * 2 * np.pi
        signal[i] = np.sin(2 * np.pi * freq * t + phase) * 0.5 
    return signal

def draw_calibration_pulse(img, start_x_idx, baseline_y_idx, interp_x, interp_y, width_grids=1, height_grids=2):
    x0 = int(interp_x(start_x_idx))
    x1 = int(interp_x(start_x_idx + width_grids))
    
    y_base = int(interp_y(baseline_y_idx))
    y_top = int(interp_y(baseline_y_idx - height_grids)) 
    
    cv2.line(img, (x0, y_base), (x0, y_top), 255, 2)
    cv2.line(img, (x0, y_top), (x1, y_top), 255, 2)
    cv2.line(img, (x1, y_top), (x1, y_base), 255, 2)

def draw_signal_segment(img, signal, time, start_x_idx, baseline_y_idx, interp_x, interp_y, time_scale_grids_per_sec=5, voltage_scale_grids_per_mv=2):
    x_indices = start_x_idx + time * time_scale_grids_per_sec
    y_indices = baseline_y_idx - signal * voltage_scale_grids_per_mv
    
    xs = interp_x(x_indices)
    ys = interp_y(y_indices)
    
    points = np.column_stack((xs, ys)).astype(np.int32)
    cv2.polylines(img, [points], isClosed=False, color=255, thickness=2)

def generate_mask_for_id(row, grid_x, grid_y, grid_df, train_dir, output_dir):
    id = str(row['id'])
    fs = row['fs']
    sig_len = row['sig_len']
    
    print(f"Processing ID: {id}")
    
    signal_df = load_signal_csv(id, train_dir)
    if signal_df is None:
        print(f"Signal not found for {id}, skipping.")
        return None, None
    
    ch0 = np.zeros((1700, 2200), dtype=np.uint8) 
    ch1 = np.zeros((1700, 2200), dtype=np.uint8) 
    ch2 = np.zeros((1700, 2200), dtype=np.uint8) 
    
    grid_point_mask = np.zeros((1700, 2200), dtype=np.uint8)
    
    interp_x = get_interp_fn(grid_x)
    interp_y = get_interp_fn(grid_y)
    inv_interp_x = get_inv_interp_fn(grid_x)
    inv_interp_y = get_inv_interp_fn(grid_y)
    
    cal_start_x_idx = inv_interp_x(79)
    row_base_ys = [708, 991, 1275, 1534]
    row_base_indices = [inv_interp_y(y) for y in row_base_ys]
    
    for base_idx in row_base_indices:
        draw_calibration_pulse(ch0, cal_start_x_idx, base_idx, interp_x, interp_y)
        
    signal_start_x_idx = inv_interp_x(118)
    
    col_start_indices = []
    for i in range(4):
        idx = signal_start_x_idx + i * 12.5
        col_start_indices.append(idx)
        
    segments = [
        (0, 0, 'I', 0.0, 2.5),
        (1, 0, 'II', 0.0, 2.5),
        (2, 0, 'III', 0.0, 2.5),
        (0, 1, 'aVR', 2.5, 5.0),
        (1, 1, 'aVL', 2.5, 5.0),
        (2, 1, 'aVF', 2.5, 5.0),
        (0, 2, 'V1', 5.0, 7.5),
        (1, 2, 'V2', 5.0, 7.5),
        (2, 2, 'V3', 5.0, 7.5),
        (0, 3, 'V4', 7.5, 10.0),
        (1, 3, 'V5', 7.5, 10.0),
        (2, 3, 'V6', 7.5, 10.0)
    ]
    
    for r_idx, c_idx, lead_name, t_start, t_end in segments:
        base_idx = row_base_indices[r_idx]
        start_x_idx = col_start_indices[c_idx]
        
        idx_start = int(t_start * fs)
        idx_end = int(t_end * fs)
        
        if idx_start >= len(signal_df) or idx_end > len(signal_df):
             pass
             
        if lead_name in signal_df.columns:
            seg_signal = signal_df[lead_name].iloc[idx_start:idx_end].values
            if np.isnan(seg_signal).any():
                seg_signal = np.nan_to_num(seg_signal)
                
            seg_time = np.arange(len(seg_signal)) / fs
            draw_signal_segment(ch0, seg_signal, seg_time, start_x_idx, base_idx, interp_x, interp_y)
            
    base_idx_long = row_base_indices[3]
    start_x_idx_long = signal_start_x_idx 
    
    if 'II' in signal_df.columns:
        long_signal = signal_df['II'].values
        long_time = np.arange(len(long_signal)) / fs
        draw_signal_segment(ch0, long_signal, long_time, start_x_idx_long, base_idx_long, interp_x, interp_y)
    
    sep_indices = [col_start_indices[1], col_start_indices[2], col_start_indices[3]]
    sep_xs = [int(interp_x(idx)) for idx in sep_indices]
    
    sep_ys_bases = [708, 991, 1274]
    
    for base_y in sep_ys_bases:
        y_top = base_y - 27
        y_bottom = base_y + 27
        for x in sep_xs:
            cv2.rectangle(ch1, (x-3, y_top), (x+3, y_bottom), 255, -1)
    
    img_height, img_width = ch2.shape
    
    for _, point in grid_df.iterrows():
        px, py = int(point['x']), int(point['y'])
        
        # Mark vertical grid lines (3 columns: x-1, x, x+1)
        for x_offset in [-1, 0, 1]:
            col_x = px + x_offset
            if 0 <= col_x < img_width:
                for row_y in range(1, img_height - 1):
                    ch2[row_y, col_x] = 255
        
        # Mark horizontal grid lines (3 rows: y-1, y, y+1)
        for y_offset in [-1, 0, 1]:
            row_y = py + y_offset
            if 0 <= row_y < img_height:
                for col_x in range(1, img_width - 1):
                    ch2[row_y, col_x] = 255
        
        # Grid point mask: mark only grid points
        if 0 <= py < img_height and 0 <= px < img_width:
            grid_point_mask[py, px] = 255
        
    mask = np.dstack((ch0, ch1, ch2))
            
    mask_dir = os.path.join(output_dir, id, 'mask')
    os.makedirs(mask_dir, exist_ok=True)
    
    mask_path = os.path.join(mask_dir, f"{id}_mask.png")
    cv2.imwrite(mask_path, mask)
    print(f"Saved mask to {mask_path}")
    
    grid_point_mask_path = os.path.join(mask_dir, f"{id}_grid_point_mask.png")
    cv2.imwrite(grid_point_mask_path, grid_point_mask)
    print(f"Saved grid point mask to {grid_point_mask_path}")
    
    return mask, grid_point_mask

def visualize(id, mask, train_dir, output_dir):
    img_path = os.path.join(train_dir, id, f"{id}-0001.png")
    if not os.path.exists(img_path):
        print(f"Original image not found: {img_path}")
        return
        
    orig_img = cv2.imread(img_path)
    overlay = np.zeros_like(orig_img)
    
    overlay[mask[:,:,0] > 0] = [0, 0, 255]
    overlay[mask[:,:,1] > 0] = [0, 255, 0]
    overlay[mask[:,:,2] > 0] = [255, 0, 0]
    
    mask_indices = np.any(overlay > 0, axis=2)
    vis_img = orig_img.copy()
    vis_img[mask_indices] = cv2.addWeighted(orig_img[mask_indices], 0.5, overlay[mask_indices], 0.5, 0)
    
    debug_dir = os.path.join(output_dir, 'debug_vis')
    os.makedirs(debug_dir, exist_ok=True)
    save_path = os.path.join(debug_dir, f"{id}_debug.jpg")
    cv2.imwrite(save_path, vis_img)
    print(f"Saved visualization to {save_path}")

def process_single_id(args):
    """Process single ID for parallel execution."""
    target_id, df_train, grid_x, grid_y, grid_df, train_dir, output_dir = args
    
    try:
        row = df_train[df_train['id'] == target_id]
        if row.empty:
            print(f"ID {target_id} not found in train.csv")
            return None
        row = row.iloc[0]
        
        mask, grid_point_mask = generate_mask_for_id(row, grid_x, grid_y, grid_df, train_dir, output_dir)
        if mask is not None:
            visualize(str(target_id), mask, train_dir, output_dir)
        return target_id
    except Exception as e:
        print(f"Error processing ID {target_id}: {e}")
        return None

def main():
    base_dir = '/Users/felix/Documents/ecg_12_1'
    train_csv = os.path.join(base_dir, 'train.csv')
    grid_csv = os.path.join(base_dir, 'grid_coordinates.csv')
    train_dir = os.path.join(base_dir, 'train')
    output_dir = os.path.join(base_dir, 'gen_train')
    
    df_train = pd.read_csv(train_csv)
    grid_x, grid_y, grid_df = load_grid_coordinates(grid_csv)
    
    target_ids = []
    if os.path.exists(train_dir):
        for item in os.listdir(train_dir):
            item_path = os.path.join(train_dir, item)
            if os.path.isdir(item_path):
                try:
                    target_ids.append(int(item))
                except ValueError:
                    print(f"Skipping non-numeric folder name: {item}")
    
    print(f"Found {len(target_ids)} IDs to process")
    
    process_args = [
        (target_id, df_train, grid_x, grid_y, grid_df, train_dir, output_dir)
        for target_id in target_ids
    ]
    
    num_processes = 12
    print(f"Starting parallel processing with {num_processes} processes...")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_id, process_args)
    
    successful = sum(1 for r in results if r is not None)
    print(f"Processing complete: {successful}/{len(target_ids)} IDs processed successfully")

if __name__ == "__main__":
    main()
