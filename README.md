# MEG-mod

**Chemically Modified siRNA Knockdown Efficiency Prediction Platform Based on Multi-view Enhanced Graph Neural Network**

---

## 📁 Project Structure

```text
project_root/
├── data_split/                # Train and test dataset splits
├── data_pre/                  # Precomputed feature files
│   ├── unimol_1b_emb_dict.pkl
│   ├── rnaernie_base_emb_fixed.pkl
│   └── cofold_results.pkl
├── Saved_Best_Models/
│   └── best_model.pt          # Trained model checkpoint
├── rnaernie/                  # RNAErnie pretrained model
├── BAN_graph.py               # Model training script
├── predict.py                 # Prediction script
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ Environment Setup

We recommend using Conda to create the running environment:

```bash
conda create -n MEG-mod python=3.10
conda activate MEG-mod
pip install -r requirements.txt
```

---

## 🔽 Download Pretrained RNAErnie Model

Please download the pretrained **RNAErnie** model from HuggingFace:

👉 https://huggingface.co/multimolecule/rnaernie  

After downloading, place the model files into the `rnaernie/` directory.

---

## 📦 Download Required Precomputed Files

Due to file size limitations, several precomputed feature files are not included in this GitHub repository.  
Before running the training pipeline, please download the following “required files” and place them into the specified directory.

### 1️⃣ Required Files

- “unimol_1b_emb_dict.pkl”  
- “rnaernie_base_emb_fixed.pkl”  
- “cofold_results.pkl”
- “best_model.pt”  

Zenodo download link:

👉 https://zenodo.org/records/18492957

---

### 2️⃣ File Placement

After downloading, organize the files as follows:

```text
project_root/
├── data_pre/
│   ├── unimol_1b_emb_dict.pkl
│   ├── rnaernie_base_emb_fixed.pkl
│   └── cofold_results.pkl
├── Saved_Best_Models/
    └── best_model.pt
```

These files contain precomputed sequence embeddings and duplex cofolding structural information required for MEG-mod.

---

## 🏋️ Model Training

Once all required files are prepared, you can train the model using:

```bash
python BAN_graph.py
```

---

## 🔍 Model Prediction

Use the trained best model to perform prediction:

```bash
python predict.py
```

---

## 📌 Notes

- Precomputed embedding and structure files are distributed separately to keep the repository lightweight.
- Ensure all required files are correctly placed before running training or inference.
- For questions regarding data or models, please contact the authors.
