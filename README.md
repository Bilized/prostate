Prostate Cancer Multimodal Analysis Framework
A comprehensive pipeline for prostate cancer prognosis and metastasis prediction using Radiomics, Deep Learning (Swin Transformer), and Clinical Data. This framework implements multi-region feature extraction, cross-attention fusion, and advanced statistical validation.

Multi-Region Radiomics: Extraction of Intra-tumoral, Core, and Peri-tumoral features using Pyradiomics.
Deep Learning Engine: Feature extraction using Swin Transformer (and ResNet/DenseNet baselines) with CBAM (Convolutional Block Attention Module).
Cross-Attention Fusion: Integration of clinical, radiomics, and deep learning features through a multi-head cross-attention mechanism.
Advanced Statistics: Comprehensive evaluation including DeLong test, NRI, IDI, Decision Curve Analysis (DCA), and Calibration curves.
Interpretability: Visual explanations via Grad-CAM and SHAP (Shapley Additive Explanations).
Survival Analysis: Kaplan-Meier (KM) curves and Cox Proportional Hazards models for risk stratification.

├── data/                       # Input CSVs and NIfTI/PNG image folders
├── models/                     # Saved .pth model weights
├── src/                        # Core Python scripts
│   ├── radiomics_extractor.py      # Feature extraction (Radiomics)
│   ├── radiomics_analysis.py       # LASSO selection & Evaluation
│   ├── swin_transformer_engine.py  # Swin-T training & feature extraction
│   ├── swin_transformer_val.py     # Swin-T external validation
│   ├── clinical_baseline.py        # Table 1 & Univariate/Multivariate Logistic
│   ├── fusion_master_pipeline.py   # Multi-modal fusion & Visualizations
│   └── survival_correlation.py     # KM, Cox, and Biomarker correlation
├── results/                    # Output plots (ROC, DCA, SHAP, etc.)
└── requirements.txt            # Python dependencies
