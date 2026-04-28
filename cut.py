import os
import numpy as np
import SimpleITK as sitk
from PIL import Image

def read_nii_file(nii_path):
    try:
        img_itk = sitk.ReadImage(nii_path)
        return sitk.GetArrayFromImage(img_itk)
    except Exception:
        return None

def find_max_tumor_slice(seg_data):
    if seg_data is None or seg_data.ndim < 3: return -1
    max_area = 0
    max_idx = -1
    # NIfTI 转为 Numpy 后维度通常是 (Depth, Height, Width)
    for i in range(seg_data.shape[0]):
        area = np.count_nonzero(seg_data[i, :, :])
        if area > max_area:
            max_area = area
            max_idx = i
    return max_idx

def normalize_image(image_slice):
    min_val, max_val = np.nanmin(image_slice), np.nanmax(image_slice)
    if max_val == min_val:
        return np.zeros_like(image_slice, dtype=np.uint8)
    img = (image_slice - min_val) / (max_val - min_val)
    return (img * 255).astype(np.uint8)

def extract_roi_with_mask(image_slice, seg_slice, padding=15):
    rows, cols = np.nonzero(seg_slice)
    if len(rows) == 0: return None, None
    
    H, W = image_slice.shape
    y_min, y_max = max(0, np.min(rows) - padding), min(H, np.max(rows) + padding)
    x_min, x_max = max(0, np.min(cols) - padding), min(W, np.max(cols) + padding)
    
    roi_img = image_slice[y_min:y_max, x_min:x_max]
    roi_mask = np.where(seg_slice[y_min:y_max, x_min:x_max] > 0, 255, 0).astype(np.uint8)
    return roi_img, roi_mask

def save_as_png(data, save_path):
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(data, mode='L').save(save_path)
    except Exception:
        pass

def process_nii_to_png(root_dirs, seq_names, save_root):
    EXPANSION_PIXELS = 15
    for root_dir in root_dirs:
        dir_name = os.path.basename(root_dir)
        if not os.path.exists(root_dir): continue
        for patient in os.listdir(root_dir):
            p_dir = os.path.join(root_dir, patient)
            if not os.path.isdir(p_dir): continue
            for seq in seq_names:
                # 兼容 .nii 和 .nii.gz
                raw_path = os.path.join(p_dir, f"{seq}.nii.gz")
                if not os.path.exists(raw_path): raw_path = raw_path.replace(".gz", "")
                
                seg_path = os.path.join(p_dir, f"{seq}_roi.nii.gz") # 假设标签命名为 seq_roi
                if not os.path.exists(seg_path): seg_path = seg_path.replace(".gz", "")
                
                if not (os.path.exists(raw_path) and os.path.exists(seg_path)): continue
                
                raw_data = read_nii_file(raw_path)
                seg_data = read_nii_file(seg_path)
                
                max_idx = find_max_tumor_slice(seg_data)
                if max_idx == -1: continue
                
                raw_slice = raw_data[max_idx, :, :]
                seg_slice = seg_data[max_idx, :, :]
                
                raw_norm = normalize_image(raw_slice)
                roi_img, roi_mask = extract_roi_with_mask(raw_norm, seg_slice, padding=EXPANSION_PIXELS)
                
                if roi_img is not None:
                    save_dir = os.path.join(save_root, dir_name, patient)
                    save_as_png(roi_img, os.path.join(save_dir, f"{seq}.png"))
                    save_as_png(roi_mask, os.path.join(save_dir, f"{seq}_mask.png"))

if __name__ == "__main__":
    root_dirs = [r"D:\NII_Data\text0", r"D:\NII_Data\text1"]
    save_root = r"D:\PNG_Output"
    seq_names = ["T2", "T2SPAIR", "DWI", "DCE"]
    process_nii_to_png(root_dirs, seq_names, save_root)
