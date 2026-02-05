# -*- coding: utf-8 -*-
# @File : BAN_graph.py

batch_SIZE = 64
epoch_NUM = 200
patience_NUM = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calc_metrics(y_true, y_pred):
    y_true_pcc = np.asarray(y_true, dtype=np.float64)
    y_pred_pcc = np.asarray(y_pred, dtype=np.float64)
    if np.isnan(y_true_pcc).any():
        idx = np.where(np.isnan(y_true_pcc))
        raise ValueError(f"y_true contains NaN at {idx}")
    if np.isnan(y_pred_pcc).any():
        idx = np.where(np.isnan(y_pred_pcc))
        raise ValueError(f"y_pred contains NaN at {idx}")
    r2 = r2_score(y_true_pcc, y_pred_pcc)
    mse = mean_squared_error(y_true_pcc, y_pred_pcc)
    mae = mean_absolute_error(y_true_pcc, y_pred_pcc)
    rmse = math.sqrt(mse)
    spcc = spearmanr(y_true_pcc, y_pred_pcc)[0]
    pcc = pearsonr(y_true_pcc, y_pred_pcc)[0]
    auc = (spcc + 1) / 2
    return [r2, mse, mae, rmse, spcc, pcc, auc]

class EarlyStopping:
    def __init__(self, patience=patience_NUM, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def build_intra_adj_edges(L: int, offset: int) -> List[Tuple[int, int]]:
    edges = []
    for i in range(L - 1):
        u = offset + i
        v = offset + i + 1
        edges.append((u, v))
        edges.append((v, u))
    return edges

def build_duplex_data(sense_seq,anti_seq,sense_x,anti_x,precomputed_pairs=None,precomputed_prob=None,use_prob=True,prob_threshold=0.2,include_intra_mfe_pairs=False):
    # print('sense_x.shape:',sense_x.shape)
    Ls, D = sense_x.shape
    La, _ = anti_x.shape
    x = torch.cat([sense_x,anti_x],dim=0)

    if precomputed_pairs is not None:
        pairs_mfe = precomputed_pairs
        prob_map = precomputed_prob or {}
    else:
        co = run_rnacofold(sense_seq, anti_seq, generate_prob=use_prob)
        pairs_mfe = dotbracket_to_pairs(co.dot_bracket)
        prob_map = parse_dotplot_ps(co.dotplot_path) if use_prob else {}

    x = torch.cat([sense_x, anti_x], dim=0)  # [Ls+La, D]

    edges: List[Tuple[int, int]] = []
    type_ids: List[int] = []
    probs: List[float] = []
    e_s = build_intra_adj_edges(Ls, 0)
    e_a = build_intra_adj_edges(La, Ls)
    edges += e_s + e_a
    type_ids += [0]*len(e_s) + [1]*len(e_a)
    probs += [1.0] * (len(e_s) + len(e_a))
    for (u, v) in pairs_mfe:
        u0, v0 = u - 1, v - 1
        is_cross = (u <= Ls and v > Ls) or (v <= Ls and u > Ls)
        p = prob_map.get((min(u, v), max(u, v)), None)
        if use_prob and (p is not None) and (p < prob_threshold):
            continue
        edges.append((u0, v0))
        edges.append((v0, u0))
        w = float(p) if (use_prob and p is not None) else 1.0
        if is_cross:
            type_ids += [2, 2]
            probs += [w, w]
        else:
            if include_intra_mfe_pairs:
                type_ids += [3, 3]
                probs += [w, w]
            else:
                edges.pop()
                edges.pop()
    if not edges:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_attr = torch.empty(0, 5, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_types = 4
    eye = torch.eye(num_types, dtype=torch.float)  # [4,4]
    type_oh = eye[torch.tensor(type_ids, dtype=torch.long)]       # [E,4]
    prob_col = torch.tensor(probs, dtype=torch.float).unsqueeze(1)  # [E,1]
    edge_attr = torch.cat([type_oh, prob_col], dim=1)               # [E,5]
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class GraphEncoder(nn.Module):
    def __init__(self, in_dim=1536, hidden=512, out_dim=1024, edge_attr_dim=5,heads=4, dropout=0.0):
        super().__init__()
        self.conv1 = TransformerConv(in_dim, hidden, heads=heads, edge_dim=edge_attr_dim,dropout=dropout,concat=False)
        self.conv2 = TransformerConv(hidden, out_dim, heads=heads, edge_dim=edge_attr_dim,dropout=dropout,concat=False)
    def forward(self, batch: Batch, return_attn: bool = False):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        if not return_attn:
            x = self.conv1(x, edge_index, edge_attr).relu()
            x = self.conv2(x, edge_index, edge_attr).relu()
            return global_mean_pool(x, batch.batch)  # [B, out_dim]
        x1, (ei1, alpha1) = self.conv1(x, edge_index, edge_attr, return_attention_weights=True)
        x1 = x1.relu()
        x2, (ei2, alpha2) = self.conv2(x1, edge_index, edge_attr, return_attention_weights=True)
        x2 = x2.relu()
        g = global_mean_pool(x2, batch.batch)
        if alpha1.dim() == 3: alpha1 = alpha1.squeeze(-1)
        if alpha2.dim() == 3: alpha2 = alpha2.squeeze(-1)
        attn = {
            "layer1": {"edge_index": ei1, "alpha": alpha1},
            "layer2": {"edge_index": ei2, "alpha": alpha2},
        }
        return g, attn

class MEG_mod_predictor(nn.Module):
    def __init__(self, device, combine_1_dim, rnaernie_dim, pc_dim,
                 use_prob=True, prob_threshold=0.2, include_intra_mfe_pairs=False):
        super().__init__()
        self.device = device
        self.max_seq_len = 27
        self.use_prob = use_prob
        self.prob_threshold = prob_threshold
        self.include_intra_mfe_pairs = include_intra_mfe_pairs
        self.phychem = {
            't': [322.21, -2.8, 4, 8, 322.05660244, 322.05660244, 146, 21, 529, 0],
            'c': [323.20, -3.4, 5, 8, 323.05185141, 323.05185141, 175, 21, 531, 0],
            'g': [363.22, -3.5, 6, 10, 363.05799942, 363.05799942, 202, 24, 598, 0],
            'a': [347.22, -3.5, 5, 11, 347.06308480, 347.06308480, 186, 23, 481, 0],
            'u': [363.22, -3.5, 6, 10, 363.05799942, 363.05799942, 202, 24, 598, 0],
        }
        with open("../../data_pre/rnaernie_base_emb_fixed.pkl", "rb") as f:
            self.base_embeddings = pickle.load(f)
        with open("../../data_pre/unimol_1b_emb_dict.pkl", "rb") as f:
            self.emb_dict = pickle.load(f)
        with open("../../data_pre/cofold_results.pkl", "rb") as f:
            self.cofold_dict = pickle.load(f)
        self.base_proj = nn.Linear(rnaernie_dim, combine_1_dim)
        self.pc_proj   = nn.Linear(pc_dim,       combine_1_dim)
        self.attn      = nn.MultiheadAttention(embed_dim=combine_1_dim, num_heads=8, batch_first=True, dropout=0.1)
        self.attn_norm = nn.LayerNorm(combine_1_dim)
        self.fused_1_proj  = nn.Linear(combine_1_dim, 1536)
        self.attention     = nn.MultiheadAttention(embed_dim=1536, num_heads=8, batch_first=True, dropout=0.1)
        self.attenion_norm = nn.LayerNorm(1536)
        self.bcn_node = weight_norm(BANLayer_token(v_dim=1536,q_dim=1536,h_dim=1536,h_out=2),name="h_mat",dim=None)
        self.bcn_mod = weight_norm(BANLayer_token(v_dim=1536, q_dim=1536, h_dim=1536, h_out=2), name='h_mat', dim=None)
        self.graph_encoder = GraphEncoder(in_dim=1536, hidden=512, out_dim=1024, heads=4, dropout=0.0)
        self.final_ban_proj = nn.Linear(3072,1024)
        self.hidden_block = nn.Sequential(
            nn.Linear(1024 + 1, 2048),
            nn.LayerNorm(2048), nn.LeakyReLU(), nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024), nn.LeakyReLU(), nn.Dropout(0.1),
        )
        self.output_block = nn.Sequential(
            nn.Linear(2048,1024),
            nn.LayerNorm(1024), nn.LeakyReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512), nn.LeakyReLU(), nn.Dropout(0.1),
            nn.Linear(512, 1),
        )
    def get_base_rnaernie_emb(self, seq_ids):
        base_dict = self.base_embeddings
        batch_list = []
        for sid in seq_ids:
            if sid not in base_dict:
                raise KeyError(f"can't find {sid} embedding")
            batch_list.append(base_dict[sid])
        batch_emb = torch.stack(batch_list, dim=0)
        return batch_emb
    def phychem_encoder(self, seq, seq_length=27, scale=100.0):
        phychem_encode = torch.zeros((seq_length, 10), dtype=torch.float, device=self.device)
        mask = torch.zeros(seq_length, dtype=torch.bool, device=self.device)
        actual_seq_len = min(len(seq), seq_length)
        for i in range(actual_seq_len):
            nt = seq[i].lower()
            if nt in self.phychem:
                vec = torch.tensor(self.phychem[nt], dtype=torch.float, device=self.device)
                phychem_encode[i, :] = vec / scale
                mask[i] = True
        return phychem_encode, mask

    def forward(self, sense_ids, anti_ids, sense_seqs, anti_seqs,
                sense_mod_types, sense_mod_positions,
                anti_mod_types, anti_mod_positions,
                concentrations):
        # 1) RNAErnie base
        sense_base_emb = self.get_base_rnaernie_emb(sense_ids).to(self.device)   # [B, L, 768]
        anti_base_emb  = self.get_base_rnaernie_emb(anti_ids).to(self.device)    # [B, L, 768]
        # 2)
        sense_encodes, sense_masks = zip(*[self.phychem_encoder(seq.lower(), seq_length=self.max_seq_len) for seq in sense_seqs])
        anti_encodes,  anti_masks  = zip(*[self.phychem_encoder(seq.lower(), seq_length=self.max_seq_len) for seq in anti_seqs])
        sense_pc  = torch.stack(sense_encodes).to(self.device)  # [B, L, 10]
        anti_pc   = torch.stack(anti_encodes).to(self.device)   # [B, L, 10]
        sense_mask = torch.stack(sense_masks).to(self.device)   # [B, L]
        anti_mask  = torch.stack(anti_masks).to(self.device)    # [B, L]
        # 3)
        sense_base_proj = self.base_proj(sense_base_emb)
        sense_pc_proj   = self.pc_proj(sense_pc)
        sense_emb, _    = self.attn(query=sense_base_proj, key=sense_pc_proj, value=sense_pc_proj, key_padding_mask=~sense_mask)
        sense_emb = sense_emb + sense_base_proj
        sense_emb       = self.attn_norm(sense_emb)             # [B, L, C]
        anti_base_proj = self.base_proj(anti_base_emb)
        anti_pc_proj   = self.pc_proj(anti_pc)
        anti_emb, _    = self.attn(query=anti_base_proj, key=anti_pc_proj, value=anti_pc_proj, key_padding_mask=~anti_mask)
        anti_emb = anti_emb + anti_base_proj
        anti_emb       = self.attn_norm(anti_emb)               # [B, L, C]
        B, L, _ = sense_emb.shape
        # 4)
        sense_pos_mod_list = []
        anti_pos_mod_list  = []
        # 用于预测
        sense_mod_tokens_list = []
        anti_mod_tokens_list  = []

        for i in range(B):
            s_types, s_pos = parse_modification_info(sense_mod_types[i], sense_mod_positions[i])
            a_types, a_pos = parse_modification_info(anti_mod_types[i],  anti_mod_positions[i])
            # 用于预测
            s_mod = generate_final_modification_embeddings(
                sense_seqs[i], s_types, s_pos, self.emb_dict, self.device, max_seq_len=self.max_seq_len)
            a_mod = generate_final_modification_embeddings(
                anti_seqs[i], a_types, a_pos, self.emb_dict, self.device, max_seq_len=self.max_seq_len)
            sense_pos_mod_list.append(s_mod)
            anti_pos_mod_list.append(a_mod)
            sense_mod_tokens_list.append(s_tokens)
            anti_mod_tokens_list.append(a_tokens)
        sense_pos_mod = torch.stack(sense_pos_mod_list,dim=0)
        anti_pos_mod = torch.stack(anti_pos_mod_list,dim=0)
        sense_mod_tokens_batch = torch.zeros(B, max_L_mod_sense, embed_dim, device=self.device)
        anti_mod_tokens_batch = torch.zeros(B, max_L_mod_anti, embed_dim, device=self.device)

        for i in range(B):
            Ls_i = sense_mod_tokens_list[i].size(0)
            sense_mod_tokens_batch[i, :Ls_i, :] = sense_mod_tokens_list[i]
            La_i = anti_mod_tokens_list[i].size(0)
            anti_mod_tokens_batch[i, :La_i, :] = anti_mod_tokens_list[i]

        sense_emb_proj = self.fused_1_proj(sense_emb)
        anti_emb_proj  = self.fused_1_proj(anti_emb)
        sense_fused_2, _ =self.bcn_node(sense_emb_proj,sense_pos_mod)
        anti_fused_2, _ = self.bcn_node(anti_emb_proj, anti_pos_mod)
        data_list = []
        for i in range(B):
            key = f"{sense_ids[i]}|{anti_ids[i]}"
            co_res = self.cofold_dict[key]
            data_i = build_duplex_data(
                sense_seqs[i], anti_seqs[i],
                sense_fused_2[i], anti_fused_2[i],
                use_prob=True,
                prob_threshold=self.prob_threshold,
                include_intra_mfe_pairs=self.include_intra_mfe_pairs,
                precomputed_pairs=co_res["pairs_mfe"],
                precomputed_prob=co_res["prob_map"]
            )
            data_list.append(data_i)
        batch_graph = Batch.from_data_list(data_list).to(self.device)
        graph_emb = self.graph_encoder(batch_graph)  # [B, 1024]

        sense_fused_mod,sense_att_mod = self.bcn_mod(sense_emb_proj,sense_mod_tokens_batch)
        anti_fused_mod,anti_att_mod = self.bcn_mod(anti_emb_proj,anti_mod_tokens_batch)
        sense_ban_seq = sense_fused_mod.mean(dim=1)  # [B, 1536]
        anti_ban_seq = anti_fused_mod.mean(dim=1)  # [B, 1536]
        ban_seq_emb = torch.cat([sense_ban_seq, anti_ban_seq], dim=-1)  # [B, 3072]
        ban_seq_emb = self.final_ban_proj(ban_seq_emb)
        combined = torch.cat([graph_emb,ban_seq_emb], dim=-1)# [B,2048]
        out = self.output_block(combined)
        return out

def evaluate(data_iter, net, criterion):
    net.eval()
    label_pred, label_true = [], []
    total_loss, num_batches = 0.0, 0

    for data in data_iter:
        sense_ids = data['sense_ids']
        anti_ids  = data['anti_ids']
        sense_seqs= data['sense_seqs']
        anti_seqs = data['anti_seqs']
        sense_mod_types = data['sense_mod_types']
        sense_mod_positions = data['sense_mod_positions']
        anti_mod_types  = data['anti_mod_types']
        anti_mod_positions = data['anti_mod_positions']
        concentrations = data['concentrations']
        pcts = data['pcts'].to(device)
        output = net(sense_ids, anti_ids, sense_seqs, anti_seqs,
                     sense_mod_types, sense_mod_positions,
                     anti_mod_types, anti_mod_positions,
                     concentrations)
        loss = criterion(output.view(-1), pcts.view(-1))
        total_loss += loss.item(); num_batches += 1
        label_true.extend(pcts.detach().cpu().numpy().flatten())
        label_pred.extend(output.squeeze().detach().cpu().numpy().flatten())
    performance = calc_metrics(label_true, label_pred)
    average_loss = total_loss / max(1, num_batches)
    return performance, average_loss, label_pred, label_true
def save_metrics(metrics, predictions, fold, epoch, data_type):
    os.makedirs(f'results/{fold}/metrics', exist_ok=True)
    os.makedirs(f'results/{fold}/predictions', exist_ok=True)
    pd.DataFrame(metrics).to_csv(f'results/{fold}/metrics/{data_type}_metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(f'results/{fold}/predictions/{data_type}_pred_epoch_{epoch + 1}.csv', index=False)

def build_warmup_cosine_scheduler(optimizer, num_training_steps, warmup_ratio=0.1, eta_min=0.0, base_lr=LEARNING_RATE):
    warmup_steps = max(1, int(warmup_ratio * num_training_steps))
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        if eta_min > 0.0:
            return (eta_min / base_lr) + (1.0 - eta_min / base_lr) * cosine
        else:
            return cosine
    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def main():

    for fold in range(1,5):
        print("-"*30 + f"k-fold: {fold+1}" + "-"*30)
        print("loading models...")
        model = MEG_mod_predictor(device=device, combine_1_dim=512, rnaernie_dim=768, pc_dim=10,
                                   use_prob=True, prob_threshold=0.2, include_intra_mfe_pairs=False).to(device)
        learning_rate = LEARNING_RATE
        weight_decay = WEIGHT_DECAY
        print("start training...")
        train_dataset = MEGDataset(f"../data_split/train_{fold + 1}.xlsx")
        val_dataset   = MEGDataset(f"../data_split/test_{fold + 1}.xlsx")
        train_loader  = DataLoader(train_dataset, batch_size=batch_SIZE, shuffle=True,  collate_fn=collate_fn)
        valid_loader  = DataLoader(val_dataset,   batch_size=batch_SIZE, shuffle=False, collate_fn=collate_fn)
        num_training_steps = epoch_NUM * len(train_loader)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.SmoothL1Loss(beta=0.5)
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            num_training_steps=num_training_steps,
            warmup_ratio=0.1,
            eta_min=0.0,
            base_lr=learning_rate
        )
        early_stopping = EarlyStopping(patience=patience_NUM)
        best_valid_pcc = -float('inf')
        lr_hist = []
        valid_metrics =  []
        valid_predictions = []
        for epoch in range(epoch_NUM):
            model.train()
            t0 = time.time()
            train_loss_ls = []
            for data in train_loader:
                sense_ids = data['sense_ids']
                anti_ids  = data['anti_ids']
                sense_seqs= data['sense_seqs']
                anti_seqs = data['anti_seqs']
                sense_mod_types = data['sense_mod_types']
                sense_mod_positions = data['sense_mod_positions']
                anti_mod_types  = data['anti_mod_types']
                anti_mod_positions = data['anti_mod_positions']
                concentrations = data['concentrations']
                pcts = data['pcts'].to(device)
                pred = model(sense_ids, anti_ids, sense_seqs, anti_seqs,
                             sense_mod_types, sense_mod_positions,
                             anti_mod_types, anti_mod_positions,
                             concentrations)
                loss = criterion(pred.view(-1), pcts.view(-1))
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_ls.append(loss.item())
                scheduler.step()
            lr_hist.append({'LR': optimizer.param_groups[0]['lr']})
            model.eval()
            with torch.no_grad():
                valid_results, valid_loss, valid_pred, valid_true = evaluate(valid_loader, model, criterion)
            print(f"\nEpoch:{epoch+1}, loss:{np.mean(train_loss_ls):.5f}, time:{time.time()-t0:.2f}\n"
                  f"Valid_R2:{valid_results[0]:.4f}|Valid_MSE:{valid_results[1]:.4f}|Valid_PCC:{valid_results[5]:.4f}|Valid_AUC:{valid_results[6]:.4f}")
            valid_metrics.append({'epoch': epoch + 1, 'R2': valid_results[0], 'MSE': valid_results[1], 'MAE': valid_results[2],
                                  'RMSE': valid_results[3], 'SPCC': valid_results[4], 'PCC': valid_results[5], 'AUC': valid_results[6],
                                  'loss': f'{valid_loss:.4f}'})
            valid_predictions.append({'pred': valid_pred, 'true': valid_true})
            valid_pcc = valid_results[5]
            if valid_pcc > best_valid_pcc:
                best_valid_pcc = valid_pcc
                os.makedirs(f'Saved_Best_Models/{fold + 1}', exist_ok=True)
                save_path_pt = f'Saved_Best_Models/{fold + 1}/best_model.pt'
                print(f'Saving model: {fold + 1}fold {epoch + 1}epoch')
                torch.save(model.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)
                early_stopping(valid_pcc, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        save_metrics(valid_metrics, valid_predictions, fold + 1, epoch, 'valid')
        lr_df = pd.DataFrame(lr_hist)
        os.makedirs(f'results/{fold + 1}/lr_list', exist_ok=True)
        lr_df.to_csv(f'results/{fold + 1}/lr_list/lr_list.csv', index=False)

if __name__ == "__main__":
    main()
