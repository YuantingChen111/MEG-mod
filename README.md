# MEG-mod
Chemically modified siRNA prediction platform based on multiview enhanced GNN

---

## 📁 Project Structure
```text
project_root/
├── data_split/              # train and test data
├── data_pre/                     
│   ├── unimol_1b_emb_dict.pkl
│   ├── rnaernie_base_emb_fixed.pkl
│   └── cofold_results.pkl
├── Save_Best_Models/
│   └── best_model.pt
├── rnaernie/
├── BAN_graph.py                # Training and prediction scripts
├── predict.py
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview and usage instructions
```

---

## ⚙️ Environment Setup
```python
conda create -n MEG-mod python=3.10
conda activate MEG-mod
pip install -r requirements.txt
```

---

## 🔽 Download Pretrained Model RNAErnie
We recommend downloading from:<br/>
+ **RNAErnie**: https://huggingface.co/multimolecule/rnaernie<br/> 
Place the downloaded models into the `rnaernie/` folder.
---

## 📦 Required Precomputed Files 
Due to file size limitations, several precomputed feature files are not included in this GitHub repository.  
Before running the code, please download the following “required files” and place them into the specified directory.

---

### 1️⃣ Download Required Files

Please download the following three files:

- “unimol_1b_emb_dict.pkl”  
- “rnaernie_base_emb_fixed.pkl”  
- “cofold_results.pkl”  

Download link (Zenodo):

👉 https://zenodo.org/records/18492957

---

### 2️⃣ File Placement

After downloading, please place all files into the “data_pre” folder under the project root directory, as shown below:


---
## 🏋️ Model Training
Due to file size limitations, several precomputed feature files are not included in this GitHub repository.  
Before running the code, please download the following “required files” and place them into the specified directory.

---

### 1️⃣ Download Required Files

Please download the following three files:

- “unimol_1b_emb_dict.pkl”  
- “rnaernie_base_emb_fixed.pkl”  
- “cofold_results.pkl”  

Download link (Zenodo):

👉 https://zenodo.org/records/18492957

---
### 2️⃣ File Placement

After downloading, please place all files into the “data_pre” folder under the project root directory, as shown below:


---
### 2️⃣ Training

```python
python BAN_graph.py
```

---

## 🔍 Model Prediction
Use the best model to make predictions on test data:
```python
python predict.py
```

---
