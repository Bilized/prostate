import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import time
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def crop_to_content(img):
    coords = cv2.findNonZero(img)
    if coords is None: return img
    x, y, w, h = cv2.boundingRect(coords)
    pad = 5
    h_i, w_i = img.shape
    return img[max(0, y - pad):min(h_i, y + h + pad), max(0, x - pad):min(w_i, x + w + pad)]


def create_zoomed_3channel(image_path):
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 0)
    if img is None: return Image.new('RGB', (224, 224))
    if cv2.countNonZero(img) > 0: img = crop_to_content(img)
    _, mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    c_r = img
    c_g = cv2.bitwise_and(img, img, mask=cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1))
    c_b = cv2.bitwise_and(img, img, mask=cv2.subtract(cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=4), mask))
    return Image.fromarray(cv2.merge([c_r, c_g, c_b]))


class ProstateDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = create_zoomed_3channel(row['Path'])
        if self.transform: img = self.transform(img)
        return img, torch.tensor(int(row['Label']), dtype=torch.long)


class SwinTransformer_Custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.num_features = self.model.head.in_features
        self.model.head = nn.Linear(self.num_features, 2)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.norm(x)
        x = self.model.permute(x)
        x = self.model.avgpool(x)
        f = self.model.flatten(x)
        return self.model.head(f), f


def evaluate_pipeline(model, tr_loader, val_loader, df_tr, df_val, device):
    model.eval()

    def get_f(loader):
        feats = []
        with torch.no_grad():
            for x, _ in loader:
                _, f = model(x.to(device))
                feats.extend(f.cpu().numpy())
        return np.array(feats)

    xt_r, xv_r = get_f(tr_loader), get_f(val_loader)
    cols = [f'S_{i}' for i in range(768)]

    dft = pd.DataFrame(xt_r, columns=cols)
    dft['PatientID'], dft['Label'] = df_tr['PatientID'].values, df_tr['Label'].values
    dft = dft.groupby('PatientID').mean().reset_index()

    dfv = pd.DataFrame(xv_r, columns=cols)
    dfv['PatientID'], dfv['Label'] = df_val['PatientID'].values, df_val['Label'].values
    dfv = dfv.groupby('PatientID').mean().reset_index()

    pipe = Pipeline([('s', StandardScaler()), ('p', PCA(n_components=0.8)),
                     ('c', LogisticRegression(C=0.001, penalty='l1', solver='liblinear'))])
    pipe.fit(dft[cols].values, dft['Label'].values)
    auc = roc_auc_score(dfv['Label'].values, pipe.predict_proba(dfv[cols].values)[:, 1])
    return auc, pd.concat([dft, dfv])


def main():
    set_seed()
    data_dir = "./data/train"
    rad_file = "./results/Radiomics_Final_Subset.csv"
    res_dir = "./results/swin_out"
    os.makedirs(res_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_rad = pd.read_csv(rad_file).sort_values('PatientID')
    tr_ids, val_ids = train_test_split(df_rad['PatientID'].unique(), test_size=0.2, random_state=42)

    imgs = []
    for l_val, l_name in [(0, "lable0"), (1, "lable1")]:
        p = os.path.join(data_dir, l_name)
        if not os.path.exists(p): continue
        for r, _, fs in os.walk(p):
            for f in fs:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    pid = os.path.basename(r)
                    imgs.append({'Path': os.path.join(r, f), 'Label': l_val, 'PatientID': pid})

    df_all = pd.DataFrame(imgs)
    df_tr = df_all[df_all['PatientID'].isin(tr_ids)]
    df_val = df_all[df_all['PatientID'].isin(val_ids)]

    t_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    v_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    tr_ld = DataLoader(ProstateDataset(df_tr, t_tf), batch_size=32, shuffle=True)
    ex_tr = DataLoader(ProstateDataset(df_tr, v_tf), batch_size=32)
    ex_val = DataLoader(ProstateDataset(df_val, v_tf), batch_size=32)

    model = SwinTransformer_Custom().to(device)
    opt = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    best_auc = 0

    for ep in range(50):
        model.train()
        for x, y in tr_ld:
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, _ = model(x.to(device))
                loss = crit(logits, y.to(device))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        if ep >= 3:
            auc, df_f = evaluate_pipeline(model, ex_tr, ex_val, df_tr, df_val, device)
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), os.path.join(res_dir, "swin_model.pth"))
                df_f.to_csv(os.path.join(res_dir, "swin_features.csv"), index=False)
            if auc > 0.9: break


if __name__ == "__main__":
    main()