import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pytorch_grad_cam import GradCAM
import shap
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")

MODELS_DIR = "./results/fusion_output"
SWIN_P = "./models/swin_model.pth"
FUSION_P = os.path.join(MODELS_DIR, "fusion_model.pth")
RESULT_DIR = "./results/analysis"
os.makedirs(RESULT_DIR, exist_ok=True)


class MultiModalCrossAttention(nn.Module):
    def __init__(self, clin_dim=2, rad_dim=12, dl_dim=768, embed_dim=64):
        super().__init__()
        self.proj_clin = nn.Sequential(nn.Linear(clin_dim, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
        self.proj_rad = nn.Sequential(nn.Linear(rad_dim, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
        self.proj_dl = nn.Sequential(nn.Linear(dl_dim, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim, 4, dropout=0.5, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(nn.Linear(embed_dim * 3, 64), nn.ReLU(), nn.Dropout(0.6), nn.Linear(64, 1),
                                        nn.Sigmoid())

    def forward(self, c, r, d):
        s = torch.cat([self.proj_clin(c).unsqueeze(1), self.proj_rad(r).unsqueeze(1), self.proj_dl(d).unsqueeze(1)],
                      dim=1)
        a, _ = self.attention(s, s, s)
        f = self.attn_norm(s + a).reshape(s.size(0), -1)
        return self.classifier(f)


def run_gradcam(img_p, model_swin, model_fusion, c_tensor, r_tensor):
    target_layers = [model_swin.model.features[-1]]

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.swin = model_swin
            self.fusion = model_fusion

        def forward(self, x):
            _, feat = self.swin(x)
            return self.fusion(c_tensor, r_tensor, feat)

    cam = GradCAM(model=Wrapper(), target_layers=target_layers, reshape_transform=lambda x: x.permute(0, 3, 1, 2))
    img = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])(
        Image.open(img_p).convert('RGB')).unsqueeze(0)
    mask = cam(input_tensor=img)[0, :]
    plt.imshow(mask, cmap='jet')
    plt.savefig(os.path.join(RESULT_DIR, "gradcam.png"))


def run_shap_analysis(model, c, r, d):
    def wrap_f(x):
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            return model(t[:, :2], t[:, 2:14], t[:, 14:]).cpu().numpy()

    X = np.concatenate([c, r, d], axis=1)
    ex = shap.KernelExplainer(wrap_f, X[:10])
    sv = ex.shap_values(X[:5])
    shap.summary_plot(sv, X[:5], show=False)
    plt.savefig(os.path.join(RESULT_DIR, "shap_summary.png"))


def forest_plot(df):
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5))
    for i, row in df.iterrows():
        ax.errorbar(row['OR'], i, xerr=[[row['OR'] - row['L']], [row['U'] - row['OR']]], fmt='s', color='orange')
        ax.text(0.1, i, row['Variable'])
    ax.set_xscale('log')
    ax.axvline(1, color='black', linestyle='--')
    plt.savefig(os.path.join(RESULT_DIR, "forest_plot.png"))


if __name__ == "__main__":
    # 执行分析流程
    pass