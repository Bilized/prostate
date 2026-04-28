import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

IN_PELVIS_DIR =
OUT_PELVIS_DIR =
SAVE_BASE_DIR =

SEQ_KEYWORDS = ["t2", "spair", "dwi", "dce"]

def read_dicom_series(directory):

    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(directory)
        if not series_ids:
            return sitk.ReadImage(directory)

        dicom_names = reader.GetGDCMSeriesFileNames(directory, series_ids[0])
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        return image
    except Exception as e:
        print(f"❌ DICOM none: {directory} | {e}")
        return None

def process_roi_dicom(src_path, save_path):
    try:
        if os.path.isdir(src_path):
            roi_itk = read_dicom_series(src_path)
        else:
            roi_itk = sitk.ReadImage(src_path)

        if roi_itk is None: return False

        roi_np = sitk.GetArrayFromImage(roi_itk)

        roi_bin = (roi_np > 0).astype(np.uint8)

        final_roi = sitk.GetImageFromArray(roi_bin)
        final_roi.CopyInformation(roi_itk)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path = os.path.splitext(save_path)[0] + ".nii.gz"
        sitk.WriteImage(final_roi, save_path)
        return True
    except Exception as e:
        print(f"⚠️ ROI pass: {e}")
        return False

def process_img_dicom(src_path, save_path):
    try:
        raw_img = read_dicom_series(src_path)
        if raw_img is None: return False
        raw_img_float = sitk.Cast(raw_img, sitk.sitkFloat32)
        try:
            mask_img = sitk.OtsuThreshold(raw_img_float, 0, 1, 200)
        except:
            mask_img = sitk.BinaryThreshold(raw_img_float, 1, 99999, 1, 0)

        try:
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations([20, 20, 20])
            corrected_img = corrector.Execute(raw_img_float, mask_img)
        except:
            corrected_img = raw_img_float

        img_arr = sitk.GetArrayFromImage(corrected_img)
        mask_arr = sitk.GetArrayFromImage(mask_img)

        pixels = img_arr[mask_arr == 1]
        if len(pixels) > 0:
            mean = np.mean(pixels)
            std = np.std(pixels)
            img_arr = (img_arr - mean) / (std + 1e-8)

        final_img = sitk.GetImageFromArray(img_arr)
        final_img.CopyInformation(raw_img)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path = os.path.splitext(save_path)[0] + ".nii.gz"
        sitk.WriteImage(final_img, save_path)
        return True

    except Exception as e:
        print(f"p: {e}")
        return False


# ================= 主程序 =================
def run_preprocessing():
    print("start")

    dirs = {'Label_0': IN_PELVIS_DIR, 'Label_1': OUT_PELVIS_DIR}
    success_count = 0
    total_count = 0

    for label_name, root_dir in dirs.items():
        if not os.path.exists(root_dir): continue

        patients = [p for p in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, p))]

        for pid in tqdm(patients, desc=f"Processing {label_name}"):
            p_src_path = os.path.join(root_dir, pid)
            p_dst_path = os.path.join(SAVE_BASE_DIR, label_name, pid)

            sub_items = os.listdir(p_src_path)

            for item in sub_items:
                item_lower = item.lower()
                full_item_path = os.path.join(p_src_path, item)
                dst_file_name = item + ".nii.gz"
                dst_file_path = os.path.join(p_dst_path, dst_file_name)

                if 'roi' in item_lower:
                    process_roi_dicom(full_item_path, dst_file_path)
                    continue

                if any(k in item_lower for k in SEQ_KEYWORDS):
                    if process_img_dicom(full_item_path, dst_file_path):
                        success_count += 1
                    total_count += 1

    print(f"finial: {success_count} / {total_count}")
    print(f"save: {SAVE_BASE_DIR}")


if __name__ == "__main__":
    run_preprocessing()