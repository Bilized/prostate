import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import os
import warnings

warnings.filterwarnings("ignore")

CLIN_PATH = "./data/clinical_data.csv"
RAD_PATH = "./results/radiomics/Radiomics_Selected_Features.csv"
RESULT_DIR = "./results/clinical"
os.makedirs(RESULT_DIR, exist_ok=True)


def run_logistic(formula, data):
    return smf.logit(formula, data=data).fit(disp=0, method='bfgs', maxiter=200)


def format_or_p(res_dict, key):
    m_key = next((k for k in res_dict.keys() if key in k or k in key), None)
    if m_key is None or np.isnan(res_dict[m_key]['OR']): return "-", "-"
    v = res_dict[m_key]
    or_str = f"{v['OR']:.2f} ({v['CI_L']:.2f}-{v['CI_U']:.2f})" if v['OR'] <= 500 else "> 500.00"
    p_str = f"**{v['P']:.3f}**" if v['P'] < 0.05 else f"{v['P']:.3f}"
    if v['P'] < 0.001: p_str = "**< 0.001**"
    return or_str, p_str


def get_custom_level_name(feat, lvl):
    l_s = str(lvl)
    if feat == 'PIRADS_Group': return "  Score ≤ 4" if lvl == 0 else "  Score 5"
    if feat == 'PSA_Group': return ["  ≤ 20", "  21-99", "  ≥ 100"][int(lvl)]
    if feat in ['ALP_Status', 'Hb_Status']: return "  Normal" if lvl == 0 else "  Abnormal"
    if feat in ['SVI', 'EPE']: return "  Absent" if l_s in ['0', '0.0', 'Absent'] else "  Present"
    return f"  Level {l_s}"


def main():
    df_rad = pd.read_csv(RAD_PATH).sort_values('PatientID')
    train_idx, _ = train_test_split(df_rad.index, test_size=0.2, stratify=df_rad.iloc[:, 1], random_state=42)
    t_ids = set(df_rad.loc[train_idx, 'PatientID'].astype(str).values)

    df_clin = pd.read_csv(CLIN_PATH)
    df_clin['PatientID'] = df_clin['PatientID'].astype(str)
    target = 'Target' if 'Target' in df_clin.columns else 'Label'
    df = df_clin[df_clin['PatientID'].isin(t_ids)].copy().dropna(subset=[target])
    df[target] = df[target].astype(int)

    act_cont = [c for c in ['Age', 'Calcium', 'Ca'] if c in df.columns]
    act_cat = [c for c in ['Gleason_ISUP', 'T_Stage', 'N_Stage', 'SVI', 'EPE', 'PIRADS'] if c in df.columns]

    df[act_cont] = SimpleImputer(strategy='median').fit_transform(df[act_cont])
    df[act_cat] = SimpleImputer(strategy='most_frequent').fit_transform(df[act_cat])

    f_cat = []
    if 'PIRADS' in act_cat:
        df['PIRADS_Group'] = df['PIRADS'].astype(float).apply(lambda x: 1 if x >= 5 else 0)
        f_cat.append('PIRADS_Group')
    if 'PSA' in df.columns:
        df['PSA_Group'] = pd.cut(df['PSA'], bins=[-np.inf, 20, 100, np.inf], labels=[0, 1, 2]).astype(int)
        f_cat.append('PSA_Group')
    for c, n in [('ALP', 'ALP_Status'), ('Hb', 'Hb_Status')]:
        if c in df.columns:
            df[n] = df[c].apply(lambda x: 0 if 40 <= x <= 150 else 1) if 'ALP' in c else df[c].apply(
                lambda x: 1 if x < 120 else 0)
            f_cat.append(n)

    uni_res, multi_res = {}, {}
    for feat in act_cont:
        r = run_logistic(f"{target} ~ {feat}", df)
        uni_res[feat] = {'OR': np.exp(r.params[feat]), 'P': r.pvalues[feat], 'CI_L': np.exp(r.conf_int().loc[feat, 0]),
                         'CI_U': np.exp(r.conf_int().loc[feat, 1])}
    for feat in f_cat:
        r = run_logistic(f"{target} ~ C({feat})", df)
        for idx in r.params.index:
            if idx != 'Intercept':
                lvl = str(idx.split('T.')[1].replace(']', ''))
                uni_res[f"{feat}_Level_{lvl}"] = {'OR': np.exp(r.params[idx]), 'P': r.pvalues[idx],
                                                  'CI_L': np.exp(r.conf_int().loc[idx, 0]),
                                                  'CI_U': np.exp(r.conf_int().loc[idx, 1])}

    m_f = f"{target} ~ " + " + ".join(act_cont + [f"C({f})" for f in f_cat])
    mr = run_logistic(m_f, df)
    for idx in mr.params.index:
        if idx != 'Intercept':
            k = f"{idx.split('C(')[1].split(')')[0]}_Level_{idx.split('T.')[1].replace(']', '')}" if "C(" in idx else idx
            multi_res[k] = {'OR': np.exp(mr.params[idx]), 'P': mr.pvalues[idx],
                            'CI_L': np.exp(mr.conf_int().loc[idx, 0]), 'CI_U': np.exp(mr.conf_int().loc[idx, 1])}

    n_neg, n_pos = sum(df[target] == 0), sum(df[target] == 1)
    rows = []
    for feat in act_cont:
        v0, v1 = df[df[target] == 0][feat], df[df[target] == 1][feat]
        u_or, u_p = format_or_p(uni_res, feat);
        m_or, m_p = format_or_p(multi_res, feat)
        rows.append(
            [f"{feat} (Med, IQR)", f"{v0.median():.1f}({np.percentile(v0, 25):.1f}-{np.percentile(v0, 75):.1f})",
             f"{v1.median():.1f}({np.percentile(v1, 25):.1f}-{np.percentile(v1, 75):.1f})", u_or, u_p, m_or, m_p])
    for feat in f_cat:
        rows.append([feat.replace('_Group', '').replace('_Status', ''), "", "", "", "", "", ""])
        lvls = sorted(df[feat].unique())
        for l in lvls:
            n0, n1 = sum((df[feat] == l) & (df[target] == 0)), sum((df[feat] == l) & (df[target] == 1))
            u_or, u_p = ("Reference", "-") if l == lvls[0] else format_or_p(uni_res, f"{feat}_Level_{l}")
            m_or, m_p = ("Reference", "-") if l == lvls[0] else format_or_p(multi_res, f"{feat}_Level_{l}")
            rows.append(
                [get_custom_level_name(feat, l), f"{n0}({(n0 / n_neg) * 100:.1f}%)", f"{n1}({(n1 / n_pos) * 100:.1f}%)",
                 u_or, u_p, m_or, m_p])

    pd.DataFrame(rows,
                 columns=["Characteristics", f"Negative(n={n_neg})", f"Positive(n={n_pos})", "Uni OR", "P", "Multi OR",
                          "P"]).to_csv(os.path.join(RESULT_DIR, "Table1_Baseline.csv"), index=False,
                                       encoding='utf-8-sig')


if __name__ == "__main__":
    main()