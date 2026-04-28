import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss, accuracy_score, recall_score, f1_score, \
    confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import scipy.stats
import warnings
import random
from math import pi

warnings.filterwarnings("ignore")

SWIN_MODEL_PATH = "./models/swin_model.pth"
RAD_TRAIN_PATH = "./data/radiomics_train.csv"
RAD_EXT_PATH = "./data/radiomics_external.csv"
CLIN_TRAIN_PATH = "./data/clinical_train.csv"
CLIN_EXT_PATH = "./data/clinical_external.csv"
TRAIN_DIR = "./data/train_images"
EXT_DIR = "./data/external_images"
RESULT_DIR = "./results/fusion_output"
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class SwinTransformer_Custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.swin_t(weights=None)
        self.model.head = nn.Linear(self.model.head.in_features, 2)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.norm(x)
        x = self.model.permute(x)
        x = self.model.avgpool(x)
        f = self.model.flatten(x)
        return self.model.head(f), f


class MultiModalCrossAttention(nn.Module):
    def __init__(self, clin_dim=2, rad_dim=12, dl_dim=768, embed_dim=64):
        super().__init__()
        self.proj_c = nn.Sequential(nn.Linear(clin_dim, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
        self.proj_r = nn.Sequential(nn.Linear(rad_dim, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
        self.proj_d = nn.Sequential(nn.Linear(dl_dim, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, 4, dropout=0.5, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.clf = nn.Sequential(nn.Linear(embed_dim * 3, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1),
                                 nn.Sigmoid())

    def forward(self, c, r, d):
        s = torch.cat([self.proj_c(c).unsqueeze(1), self.proj_r(r).unsqueeze(1), self.proj_d(d).unsqueeze(1)], dim=1)
        o, _ = self.attn(s, s, s)
        f = self.norm(s + o).reshape(s.size(0), -1)
        return self.clf(f).squeeze(1)


def scan_imgs(root_dir, ids):
    l = []
    for d, lb in [('label0', 0), ('label1', 1)]:
        p = os.path.join(root_dir, d)
        if not os.path.exists(p): continue
        for r, _, fs in os.walk(p):
            pid = os.path.basename(r)
            if pid in ids:
                for f in fs:
                    if f.lower().endswith(('.png', '.jpg')): l.append((os.path.join(r, f), lb, pid))
    return l


def extract_dl_features(img_list, ids, model, tf):
    ld = DataLoader(TensorDataset(torch.arange(len(img_list))), batch_size=32)
    feats, pids = [], []
    with torch.no_grad():
        for i in ld:
            batch_p = [img_list[idx][0] for idx in i[0]]
            batch_pid = [img_list[idx][2] for idx in i[0]]
            imgs = torch.stack([tf(Image.open(p).convert('RGB')) for p in batch_p]).to(DEVICE)
            _, f = model(imgs)
            feats.extend(f.cpu().numpy());
            pids.extend(batch_pid)
    df = pd.DataFrame(feats);
    df['PatientID'] = pids
    df = df.groupby('PatientID').mean().reset_index()
    return pd.merge(pd.DataFrame({'PatientID': list(ids)}), df, on='PatientID', how='left').drop(
        columns='PatientID').fillna(0).values


def main():
    set_seed()
    tr_r = pd.read_csv(RAD_TRAIN_PATH)
    ex_r = pd.read_csv(RAD_EXT_PATH)
    tr_ids, val_ids = train_test_split(tr_r['PatientID'].astype(str).unique(), test_size=0.2, random_state=42)

    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    swin = SwinTransformer_Custom().to(DEVICE)
    swin.load_state_dict(torch.load(SWIN_MODEL_PATH))
    swin.eval()

    dl_tr = extract_dl_features(scan_imgs(TRAIN_DIR, set(tr_ids)), set(tr_ids), swin, tf)
    dl_v = extract_dl_features(scan_imgs(TRAIN_DIR, set(val_ids)), set(val_ids), swin, tf)
    dl_ex = extract_dl_features(scan_imgs(EXT_DIR, set(ex_r['PatientID'].astype(str))),
                                set(ex_r['PatientID'].astype(str)), swin, tf)

    model = MultiModalCrossAttention(clin_dim=2, rad_dim=tr_r.shape[1] - 2).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.BCELoss()

    # ... (此处省略重复的训练循环逻辑，直接进入推理与统计)

    torch.save(model.state_dict(), os.path.join(RESULT_DIR, "fusion_model.pth"))


if __name__ == "__main__":
    main()