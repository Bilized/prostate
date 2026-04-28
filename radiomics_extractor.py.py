import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "./data/preprocessed"
CLINICAL_FILE = "./data/clinical_data.csv"
RESULT_DIR = "./results/extracted_features"
SAVE_FILENAME = "Radiomics_Features_Raw.csv"

SEQ_KEYWORDS = ["t2", "spair", "dwi", "dce"]
EROSION_RADIUS = 3
DILATION_RADIUS = 5

class MultiregionRadiomics:
    def __init__(self):
        settings = {
            'binWidth': 25,
            'interpolator': sitk.sitkBSpline,
            'resampledPixelSpacing': [1, 1],
            'force2D': True,
            'normalize': True,
            'normalizeScale': 100
        }
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        self.extractor.enableAllImageTypes()
        self.extractor.enableAllFeatures()

    def get_masks(self, roi_img):
        masks = {'Tumor': roi_img}
        eroder = sitk.BinaryErodeImageFilter()
        eroder.SetKernelRadius(EROSION_RADIUS)
        masks['Core'] = eroder.Execute(roi_img)
        dilater = sitk.BinaryDilateImageFilter()
        dilater.SetKernelType(sitk.sitkBall)
        dilater.SetKernelRadius(DILATION_RADIUS)
        expanded_roi = dilater.Execute(roi_img)
        masks['Peri'] = sitk.SubtractImageFilter().Execute(expanded_roi, roi_img)
        return masks

    def process_patient(self, img_path, roi_path):
        try:
            img, roi = sitk.ReadImage(img_path), sitk.ReadImage(roi_path)
            mask_dict = self.get_masks(roi)
            feats = {}
            for region, mask in mask_dict.items():
                if sitk.LabelStatisticsImageFilter().Execute(roi, mask).GetCount(1) < 5: continue
                res = self.extractor.execute(img, mask)
                for k, v in res.items():
                    if any(x in k for x in ['original_', 'wavelet_', 'log_']):
                        feats[f"{region}_{k}"] = v
            return feats
        except: return None

def main():
    if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR, exist_ok=True)
    engine = MultiregionRadiomics()
    results = []

    for label_val, label_name in [(0, "Label_0"), (1, "Label_1")]:
        folder_path = os.path.join(DATA_DIR, label_name)
        if not os.path.exists(folder_path): continue
        patients = [p for p in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, p))]

        for pid in tqdm(patients, desc=f"Extracting {label_name}"):
            p_path = os.path.join(folder_path, pid)
            pat_data = {'PatientID': pid, 'Label': label_val}
            valid, files = False, os.listdir(p_path)
            for seq in SEQ_KEYWORDS:
                img_f = next((f for f in files if seq in f.lower() and "roi" not in f.lower() and f.endswith('.nii.gz')), None)
                roi_f = next((f for f in files if f"{seq}_roi" in f.lower() and f.endswith('.nii.gz')), None)
                if img_f and roi_f:
                    feats = engine.process_patient(os.path.join(p_path, img_f), os.path.join(p_path, roi_f))
                    if feats:
                        valid = True
                        for k, v in feats.items(): pat_data[f"{seq}_{k}"] = v
            if valid: results.append(pat_data)

    if results:
        df_final = pd.DataFrame(results)
        df_final.to_csv(os.path.join(RESULT_DIR, SAVE_FILENAME), index=False)

if __name__ == "__main__":
    main()