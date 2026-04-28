import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import spearmanr, mannwhitneyu, chi2
from sklearn.metrics import accuracy_score, roc_curve
import warnings

warnings.filterwarnings("ignore")

SURVIVAL_DATA = "./data/survival.csv"
RISK_SCORE_DATA = "./results/fusion/Table_Patient_Risk_Stratification.csv"
CLINICAL_DATA = "./data/clinical_data.csv"
CD31_DATA = "./data/CD31_MVD.csv"
RESULT_DIR = "./results/survival_correlation"
os.makedirs(RESULT_DIR, exist_ok=True)


def plot_km_curves(df, time_col, event_col, group_col, save_p):
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))
    for g in df[group_col].unique():
        mask = (df[group_col] == g)
        kmf.fit(df[time_col][mask], df[event_col][mask], label=g)
        kmf.plot_survival_function()
    p_val = logrank_test(df[time_col][df[group_col] == df[group_col].unique()[0]],
                         df[time_col][df[group_col] == df[group_col].unique()[1]],
                         df[event_col][df[group_col] == df[group_col].unique()[0]],
                         df[event_col][df[group_col] == df[group_col].unique()[1]]).p_value
    plt.title(f"Log-rank P: {p_val:.4f}")
    plt.savefig(save_p)
    plt.close()


def run_cox_analysis(df, time_col, event_col, features):
    cph = CoxPHFitter()
    cols = [time_col, event_col] + features
    df_cox = df[cols].dropna()
    cph.fit(df_cox, duration_col=time_col, event_col=event_col)
    cph.summary.to_csv(os.path.join(RESULT_DIR, "Cox_Summary.csv"))
    return cph


def analyze_correlation(df, x_col, y_col, save_p):
    corr, p = spearmanr(df[x_col], df[y_col])
    plt.figure(figsize=(6, 6))
    sns.regplot(x=x_col, y=y_col, data=df, scatter_kws={'alpha': 0.5})
    plt.title(f"Spearman r: {corr:.3f}, P: {p:.3f}")
    plt.savefig(save_p)
    plt.close()


def mcnemar_test(y_true, pred1, pred2):
    c1 = (pred1 == y_true).astype(int)
    c2 = (pred2 == y_true).astype(int)
    b = np.sum((c1 == 1) & (c2 == 0))
    c = np.sum((c1 == 0) & (c2 == 1))
    if b + c == 0: return 1.0
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    return chi2.sf(stat, 1)


def main():
    if not os.path.exists(SURVIVAL_DATA) or not os.path.exists(RISK_SCORE_DATA): return

    df_s = pd.read_csv(SURVIVAL_DATA)
    df_r = pd.read_csv(RISK_SCORE_DATA)
    df = pd.merge(df_s, df_r, on='PatientID')

    plot_km_curves(df, 'Time', 'Event', 'Risk_Group', os.path.join(RESULT_DIR, "KM_Plot.png"))

    run_cox_analysis(df, 'Time', 'Event', ['Age', 'PSA', 'MF_DLM_Score'])

    if os.path.exists(CD31_DATA):
        df_c = pd.read_csv(CD31_DATA)
        df_corr = pd.merge(df, df_c, on='PatientID')
        analyze_correlation(df_corr, 'CD31_MVD', 'MF_DLM_Score', os.path.join(RESULT_DIR, "CD31_Correlation.png"))

    y_true = df['True_Label']
    p_ai = (df['MF_DLM_Score'] >= 0.5).astype(int)
    p_clin = (df['Clin_Score'] >= 0.5).astype(int)
    p_val = mcnemar_test(y_true, p_ai, p_clin)
    print(f"McNemar Test P-value: {p_val:.4f}")


if __name__ == "__main__":
    main()