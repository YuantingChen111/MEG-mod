# -*- coding: utf-8 -*-
# @File : utils.py

import torch
import os
import re
import math
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

def get_standard_embedding(mod_name, emb_dict, device, embed_dim=1536):
    if mod_name in emb_dict and emb_dict[mod_name] is not None:
        return torch.tensor(emb_dict[mod_name], dtype=torch.float, device=device)
    else:
        return torch.zeros(embed_dim, dtype=torch.float, device=device)
def get_sequence_standard_embeddings(sequence, emb_dict, device, embed_dim=1536, max_seq_len=27):

    standard_emb_names = {
        'sugar': 'Standard sugar',
        'phosphate': 'Standard phosphate',
        'base': {
            'a': 'Standard adenine',
            't': 'Standard thymine',
            'u': 'Standard uracil',
            'c': 'Standard cytosine',
            'g': 'Standard guanine'
        }
    }
    standard_embeddings = torch.zeros(max_seq_len, embed_dim, dtype=torch.float, device=device)
    standard_sugar_emb = get_standard_embedding(standard_emb_names['sugar'], emb_dict, device, embed_dim)
    standard_phosphate_emb = get_standard_embedding(standard_emb_names['phosphate'], emb_dict, device, embed_dim)
    actual_seq_len = min(len(sequence), max_seq_len)
    for i in range(actual_seq_len):
        nucleotide = sequence[i]
        if nucleotide.lower() in standard_emb_names['base']:
            standard_base_emb = get_standard_embedding(
                standard_emb_names['base'][nucleotide.lower()], emb_dict, device, embed_dim)
        else:
            standard_base_emb = torch.zeros(embed_dim, dtype=torch.float, device=device)
        standard_embeddings[i] = standard_sugar_emb + standard_base_emb + standard_phosphate_emb
    return standard_embeddings

def parse_modification_info(mod_types_str, mod_positions_str):

    if pd.isna(mod_types_str) or mod_types_str == "" or mod_types_str == "None":
        return [], []
    mod_types = [mod_type.strip() for mod_type in mod_types_str.split('*')] if isinstance(mod_types_str, str) else []
    mod_positions = []
    if pd.isna(mod_positions_str) or mod_positions_str == "" or mod_positions_str == "None":
        mod_positions = [[] for _ in mod_types]
    else:
        pos_strs = mod_positions_str.split('*') if isinstance(mod_positions_str, str) else []
        for pos_str in pos_strs:
            if pos_str.strip():
                positions = [int(p.strip()) for p in pos_str.split(',') if p.strip()]
                mod_positions.append(positions)
            else:
                mod_positions.append([])
    return mod_types, mod_positions
def get_modification_embedding(mod_name, emb_dict, device, embed_dim=1536):
    if mod_name in emb_dict and emb_dict[mod_name] is not None:
        return torch.tensor(emb_dict[mod_name], dtype=torch.float, device=device)
    else:
        return torch.zeros(embed_dim, dtype=torch.float, device=device)
def generate_position_modification_embeddings(sequence, mod_types, mod_positions, emb_dict, device, embed_dim=1536,
                                              max_seq_len=27):

    position_mod_embeddings = torch.zeros(max_seq_len, embed_dim, dtype=torch.float, device=device)
    for mod_type, positions in zip(mod_types, mod_positions):
        if not positions:
            continue

        mod_emb = get_modification_embedding(mod_type, emb_dict, device, embed_dim)
        for pos in positions:
            array_index = pos - 1
            if 0 <= array_index < max_seq_len:
                position_mod_embeddings[array_index] += mod_emb
    return position_mod_embeddings
def generate_final_modification_embeddings(sequence, mod_types, mod_positions, emb_dict, device, embed_dim=1536,
                                           max_seq_len=27):
    standard_embeddings = get_sequence_standard_embeddings(sequence.lower(), emb_dict, device, embed_dim, max_seq_len)
    modification_embeddings = generate_position_modification_embeddings(
        sequence, mod_types, mod_positions, emb_dict, device, embed_dim, max_seq_len)
    final_mod_embeddings = standard_embeddings + modification_embeddings
    return final_mod_embeddings

@dataclass
class CofoldResult:
    dot_bracket: str
    mfe_energy: float
    dotplot_path: Optional[str]

def run_rnacofold(sense: str, antisense: str, generate_prob: bool = True, temperature: Optional[float] = None) -> CofoldResult:
    cmd = ["RNAcofold", "--noPS"]
    if generate_prob:
        cmd.append("-p")
    if temperature is not None:
        cmd += ["-T", str(float(temperature))]
    if shutil.which("RNAcofold") is None:
        raise RuntimeError("RNAcofold can't be found in PATH")
    with tempfile.TemporaryDirectory() as tmpd:
        inp = f">seq\n{sense}&{antisense}\n".encode()
        try:
            res = subprocess.run(cmd, input=inp, cwd=tmpd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"RNAcofold failed\nSTDOUT:\n{e.stdout.decode(errors='ignore')}\nSTDERR:\n{e.stderr.decode(errors='ignore')}"
            )
        stdout = res.stdout.decode(errors="ignore")
        dot_bracket, mfe = parse_cofold_stdout(stdout)
        dotplot_path = None
        if generate_prob:
            candidates = [os.path.join(tmpd, n) for n in os.listdir(tmpd) if n.endswith("_dp.ps") or n == "dot.ps" or n.endswith('.ps')]
            if candidates:
                candidates.sort(key=lambda p: ("_dp.ps" not in p, os.path.getsize(p) if os.path.exists(p) else 0))
                src = candidates[0]
                fd, new_path = tempfile.mkstemp(suffix=".ps", prefix="cofold_dp_")
                os.close(fd)
                shutil.copyfile(src, new_path)
                dotplot_path = new_path
        return CofoldResult(dot_bracket, mfe, dotplot_path)
def parse_cofold_stdout(stdout: str) -> Tuple[str, float]:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    pat = re.compile(r"([().]+)&([().]+)\s*\(([-+]?\d+\.?\d*)\)")
    for ln in reversed(lines):
        m = pat.search(ln)
        if m:
            return f"{m.group(1)}&{m.group(2)}", float(m.group(3))
    pat2 = re.compile(r"([().]+)&([().]+)")
    for ln in reversed(lines):
        m = pat2.search(ln)
        if m:
            return f"{m.group(1)}&{m.group(2)}", float("nan")
    raise ValueError("can't predict dot-bracket")
def dotbracket_to_pairs(db: str) -> List[Tuple[int, int]]:
    if "&" in db:
        left, right = db.split("&", 1)
        Ls = len(left)
        seq_db = left + right
    else:
        Ls = len(db)
        seq_db = db
    stack, pairs = [], []
    for idx, ch in enumerate(seq_db, start=1):
        if ch == '(':
            stack.append(idx)
        elif ch == ')':
            if not stack:
                continue
            u = stack.pop(); v = idx
            if u < v:
                pairs.append((u, v))
            else:
                pairs.append((v, u))
    pairs.sort()
    return pairs

def parse_dotplot_ps(ps_path: str) -> Dict[Tuple[int, int], float]:
    if ps_path is None or not os.path.exists(ps_path):
        return {}
    prob_map: Dict[Tuple[int, int], float] = {}
    with open(ps_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4 and parts[3] in ("ubox", "lbox"):
                try:
                    i = int(parts[0]); j = int(parts[1]); sqrtp = float(parts[2])
                except Exception:
                    continue
                if i == j:
                    continue
                u, v = (i, j) if i < j else (j, i)
                p = sqrtp * sqrtp
                if (u, v) not in prob_map or p > prob_map[(u, v)]:
                    prob_map[(u, v)] = p
    return prob_map

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()
        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BANLayer_token(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3): #k是最后sumpooling时的stride=3
        super(BANLayer_token, self).__init__()

        self.c = 32
        self.k = k  # 3
        self.v_dim = v_dim  # 128
        self.q_dim = q_dim  # 128
        self.h_dim = h_dim  # 128#
        self.h_out = h_out  # 2

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)
        self.ln = nn.LayerNorm(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bvk', (v, att_map, q))
        if self.k > 1:
            # sum pooling
            B, v_num, hk = fusion_logits.shape
            fusion_logits = fusion_logits.view(B, v_num, self.h_dim, self.k).sum(dim=3)
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])  # [batch, v_num, hidden]
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])  # [batch, v_num, hidden]
            logits += logits_i

        logits = self.ln(logits)  # [batch, v_num, hidden]
        return logits, att_maps

def normalize_positions_to_int_list(pos):

    if pos is None:
        return []
    out = []

    def _push(x):
        if x is None:
            return
        # numpy scalar
        try:
            import numpy as np
            if isinstance(x, np.generic):
                x = x.item()
        except Exception:
            pass

        # str
        if isinstance(x, str):
            x = x.strip()
            if not x:
                return
            if "," in x:
                for t in x.split(","):
                    t = t.strip()
                    if t:
                        out.append(int(float(t)))
                return
            out.append(int(float(x)))
            return

        # int/float
        if isinstance(x, (int, float)):
            if isinstance(x, float) and (x != x):  # nan
                return
            out.append(int(x))
            return

        # list/tuple/set
        if isinstance(x, (list, tuple, set)):
            for y in x:
                _push(y)
            return
        out.append(int(x))
    _push(pos)
    out = sorted(set(out))
    return out

def flatten_and_zero_base(pos):
    if not pos:
        return []

    out = []
    for p in pos:
        if isinstance(p, (list, tuple)):
            out.extend([int(x) - 1 for x in p])
        else:
            out.append(int(p) - 1)
    return out