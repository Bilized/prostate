import pandas as pd
import numpy as np
import os, warnings, re, shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso, LogisticRegression
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, roc_curve, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

TRAIN_RAW_PATH = "./results/extracted_features/Radiomics_Features_Raw.csv"
EXTERNAL_RAW_PATH = "./data/external_features.csv"
RESULT_DIR = "./results/analysis"

for sub in ['ROC', 'CM', 'Radar', 'Tables', 'Interpretability']:
    os.makedirs(os.path.join(RESULT_DIR, sub), exist_ok=True)

def clean_feature_names(df):
    corrections = {'girIm': 'glrlm', 'Girim': 'glrlm', 'glszm': 'glszm', 'Glszm': 'glszm'}
    new_cols = []
    for col in df.columns:
        temp = col
        for w, r in corrections.items():
            if w in temp: temp = temp.replace(w, r)
        temp = re.sub(r"\[|\]|<", "_", temp)
        new_cols.append(temp)
    df.columns = new_cols
    return df

def run_lasso_selection(df):
    label_col = next(c for c in df.columns if 'label' in c.lower())
    X = df.drop(columns=[df.columns[0], label_col]).select_dtypes(include=[np.number])
    y = df[label_col].values
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(SimpleImputer(strategy='median').fit_transform(X_train))
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
    lasso_cv = LassoCV(cv=cv, max_iter=5000, random_state=42).fit(X_train_scaled, y_train)
    mse_mean = np.mean(lasso_cv.mse_path_, axis=1)
    idx_min = np.argmin(mse_mean)
    mse_threshold = mse_mean[idx_min] + np.std(lasso_cv.mse_path_[idx_min,:])/np.sqrt(5)
    alpha_1se = next(lasso_cv.alphas_[i] for i in range(len(mse_mean)) if mse_mean[i] <= mse_threshold)
    model = Lasso(alpha=alpha_1se).fit(X_train_scaled, y_train)
    selected_feats = X.columns[model.coef_ != 0].tolist()
    return selected_feats

def train_and_eval(X_train, y_train, X_val, y_val):
    models = {
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42),
        'MLP': MLPClassifier(max_iter=1000, random_state=42)
    }
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    perf_results = []
    for name, clf in models.items():
        clf.fit(X_res, y_res)
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        perf_results.append({'Model': name, 'Val_AUC': auc})
    return pd.DataFrame(perf_results), models['Logistic Regression']

def plot_shap_summary(model, X_scaled):
    explainer = shap.LinearExplainer(model, X_scaled)
    shap_values = explainer.shap_values(X_scaled)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_scaled, show=False)
    plt.savefig(os.path.join(RESULT_DIR, 'Interpretability', 'SHAP_Summary.png'), dpi=300)
    plt.close()

def main():
    if not os.path.exists(TRAIN_RAW_PATH): return
    df_train = clean_feature_names(pd.read_csv(TRAIN_RAW_PATH))
    selected_feats = run_lasso_selection(df_train)
    label_col = next(c for c in df_train.columns if 'label' in c.lower())
    X, y = df_train[selected_feats], df_train[label_col]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    performance, lr_model = train_and_eval(X_train_s, y_train, X_val_s, y_val)
    plot_shap_summary(lr_model, pd.DataFrame(X_train_s, columns=selected_feats))

if __name__ == "__main__":
    main()