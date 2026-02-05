# -*- coding: utf-8 -*-
# @File    : predict.py

import os
import pickle
import torch
import pandas as pd
import numpy as np
from typing import Dict, List
from multimolecule import RnaTokenizer, RnaErnieModel
from BAN_graph import (
    MEG_mod_predictor,
    run_rnacofold,
    dotbracket_to_pairs,
    parse_dotplot_ps
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNAERNIE_DIR = "../../rnaernie"
BASE_PKL = "../../data_pre/rnaernie_base_emb_fixed.pkl"
COFOLD_PKL = "../../data_pre/cofold_results.pkl"
UNIMOL_PKL = "../../data_pre/unimol_1b_emb_dict.pkl"

CKPT_PATH = "Saved_Best_Models/best_model.pt"
PRED_INPUT_PATH = "your_data"  # 文件 or 文件夹
OUT_DIR = "./predict_results"
USE_PROB = True
PROB_THRESHOLD = 0.2
INCLUDE_INTRA_MFE = False
BATCH_SIZE = 64
SEQ_LEN = 27
MAX_LEN = 29

def load_rnaernie(device):
    tokenizer = RnaTokenizer.from_pretrained(RNAERNIE_DIR)
    model = RnaErnieModel.from_pretrained(RNAERNIE_DIR).to(device)
    model.eval()
    return tokenizer, model
@torch.no_grad()
def compute_base_emb(seqs, ids, tokenizer, model):
    out = {}
    for sid, seq in zip(ids, seqs):
        sid = str(sid)
        token = tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=29
        )
        token = {k: v.to(DEVICE) for k, v in token.items()}
        h = model(**token,output_hidden_states=True).last_hidden_state[:, 1:-1, :]  # [1,L,768]
        # h = h[0].cpu()
        out[sid] = h[0].cpu()
    return out
def ensure_base_embeddings(df, base_dict):
    need_ids, need_seqs = [], []
    for _, row in df.iterrows():
        for k, s in [("sense_id", "sense"), ("anti_id", "antisense")]:
            sid = str(row[k])
            if sid not in base_dict:
                need_ids.append(sid)
                need_seqs.append(str(row[s]))

    if need_ids:
        tokenizer, model = load_rnaernie(DEVICE)
        new_embs = compute_base_emb(need_seqs, need_ids, tokenizer, model)
        base_dict.update(new_embs)
        print(f"[RNAErnie] {len(new_embs)} new embedding")
    return base_dict

def ensure_cofold(df, cofold_dict):
    for _, row in df.iterrows():
        key = f"{row['sense_id']}|{row['anti_id']}"
        if key not in cofold_dict:
            co = run_rnacofold(row["sense"], row["antisense"], generate_prob=USE_PROB)
            pairs = dotbracket_to_pairs(co.dot_bracket)
            prob_map = parse_dotplot_ps(co.dotplot_path) if USE_PROB else {}
            cofold_dict[key] = {
                "dot_bracket": co.dot_bracket,
                "pairs_mfe": pairs,
                "prob_map": prob_map
            }
            print(f"[cofold] new {key}")
    return cofold_dict

def predict_one_file(df: pd.DataFrame, out_path: str):
    with open(BASE_PKL, "rb") as f:
        base_dict = pickle.load(f)
    with open(COFOLD_PKL, "rb") as f:
        cofold_dict = pickle.load(f)
    base_dict = ensure_base_embeddings(df, base_dict)
    cofold_dict = ensure_cofold(df, cofold_dict)
    model = MEG_mod_predictor(
        device=DEVICE,
        combine_1_dim=512,
        rnaernie_dim=768,
        pc_dim=10,
        use_prob=USE_PROB,
        prob_threshold=PROB_THRESHOLD,
        include_intra_mfe_pairs=INCLUDE_INTRA_MFE,
    ).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.base_embeddings = base_dict
    model.cofold_dict = cofold_dict
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(df), BATCH_SIZE):
            sub = df.iloc[i:i+BATCH_SIZE]
            out = model(
                sub["sense_id"].astype(str).tolist(),
                sub["anti_id"].astype(str).tolist(),
                sub["sense"].astype(str).tolist(),
                sub["antisense"].astype(str).tolist(),
                sub["sense_mod_types"].astype(str).tolist(),
                sub["sense_mod_positions"].astype(str).tolist(),
                sub["anti_mod_types"].astype(str).tolist(),
                sub["anti_mod_positions"].astype(str).tolist(),
                sub["concentration"].tolist(),
            )
            preds.extend(out.view(-1).cpu().numpy().tolist())

    df_out = df.copy()
    df_out["pred"] = preds
    df_out.to_csv(out_path, index=False)

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    predict_path = "your_data"
    out_name = os.path.splitext(os.path.basename(predict_path))[0] + "_pred.csv"
    out_path = os.path.join(OUT_DIR, out_name)
    predict_one_file(predict_path, out_path)
