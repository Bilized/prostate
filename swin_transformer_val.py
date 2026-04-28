import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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


class ProstateDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return self.transform(create_zoomed_3channel(row['Path'])), torch.tensor(int(row['Label']), dtype=torch.long)


def extract_features(df, model, device, tf):
    ld = DataLoader(ProstateDataset(df, tf), batch_size=32)
    feats = []
    with torch.no_grad():
        for x, _ in ld:
            _, f = model(x.to(device))
            feats.extend(f.cpu().numpy())
    cols = [f'S_{i}' for i in range(768)]
    df_res = pd.DataFrame(feats, columns=cols)
    df_res['PatientID'], df_res['Label'] = df['PatientID'].values, df['Label'].values
    return df_res.groupby('PatientID').mean().reset_index(), cols


def main():
    model_p = "./results/swin_out/swin_model.pth"
    ext_dir = "./data/external"
    train_dir = "./data/train"
    rad_p = "./results/Radiomics_Final_Subset.csv"
    out_p = "./results/external_val"
    os.makedirs(out_p, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinTransformer_Custom().to(device)
    model.load_state_dict(torch.load(model_p))
    model.eval()

    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def scan(p_0, p_1):
        l = []
        for d, lb in [(p_0, 0), (p_1, 1)]:
            for r, _, fs in os.walk(d):
                for f in fs:
                    if f.lower().endswith(('.png', '.jpg')):
                        l.append({'Path': os.path.join(r, f), 'Label': lb, 'PatientID': os.path.basename(r)})
        return pd.DataFrame(l)

    df_rad = pd.read_csv(rad_p)
    tr_ids, _ = train_test_split(df_rad['PatientID'].unique(), test_size=0.2, random_state=42)
    df_train = scan(os.path.join(train_dir, "lable0"), os.path.join(train_dir, "lable1"))
    df_train = df_train[df_train['PatientID'].isin(tr_ids)]
    df_ext = scan(os.path.join(ext_dir, "text0"), os.path.join(ext_dir, "text1"))

    xt, cols = extract_features(df_train, model, device, tf)
    xe, _ = extract_features(df_ext, model, device, tf)

    pipe = Pipeline([('s', StandardScaler()), ('p', PCA(n_components=0.8)),
                     ('c', LogisticRegression(C=0.001, penalty='l1', solver='liblinear'))])
    pipe.fit(xt[cols].values, xt['Label'].values)
    probs = pipe.predict_proba(xe[cols].values)[:, 1]

    pd.DataFrame({'PatientID': xe['PatientID'], 'True': xe['Label'], 'Score': probs}).to_csv(
        os.path.join(out_p, "ext_scores.csv"), index=False)

    fpr, tpr, _ = roc_curve(xe['Label'], probs)
    plt.plot(fpr, tpr, label=f'Swin (AUC={roc_auc_score(xe["Label"], probs):.3f})')
    plt.legend()
    plt.savefig(os.path.join(out_p, "ext_roc.png"))


if __name__ == "__main__":
    main()