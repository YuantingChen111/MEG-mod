# -*- coding: utf-8 -*-
# @File : dataset_pre.py


import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MEGDataset(Dataset):
    def __init__(self, excel_path, max_seq_len=27):
        self.max_seq_len = max_seq_len
        self.df = pd.read_excel(excel_path)
        self._clean_data()
    def _clean_data(self):
        self.df = self.df.dropna(subset=['sense', 'antisense', 'knockdown'])
        self.df['sense'] = self.df['sense'].astype(str)
        self.df['antisense'] = self.df['antisense'].astype(str)
        self.df['knockdown'] = pd.to_numeric(self.df['knockdown'], errors='coerce')
        self.df['concentration'] = pd.to_numeric(self.df['concentration'], errors='coerce')
        self.df = self.df.dropna(subset=['concentration'])
        self.df = self.df.dropna(subset=['knockdown'])
        for col in ['modification_sense', 'modification_antisense', 'sense_position', 'antisense_position']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str)
                self.df[col] = self.df[col].replace('nan', None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        sense_id = row['sense_id']
        anti_id = row['anti_id']
        sense_seq = str(row['sense']).lower().strip()
        anti_seq = str(row['antisense']).lower().strip()
        pct = float(row['knockdown'])/100.0
        concentration = float(row['concentration'])
        sense_mod_types = row.get('modification_sense', '')
        sense_mod_positions = row.get('sense_position', '')
        anti_mod_types = row.get('modification_antisense', '')
        anti_mod_positions = row.get('antisense_position', '')
        sense_mod_types = sense_mod_types if pd.notna(sense_mod_types) else ''
        sense_mod_positions = sense_mod_positions if pd.notna(sense_mod_positions) else ''
        anti_mod_types = anti_mod_types if pd.notna(anti_mod_types) else ''
        anti_mod_positions = anti_mod_positions if pd.notna(anti_mod_positions) else ''
        return {
            'sense_id': sense_id,
            'anti_id': anti_id,
            'sense_seq': sense_seq,
            'anti_seq': anti_seq,
            'sense_mod_types': sense_mod_types,
            'sense_mod_positions': sense_mod_positions,
            'anti_mod_types': anti_mod_types,
            'anti_mod_positions': anti_mod_positions,
            'concentration':concentration,
            'pct': pct
        }
    def get_batch_data(self, indices):
        batch_data = {
            'sense_ids': [],
            'anti_ids': [],
            'sense_seqs': [],
            'anti_seqs': [],
            'sense_mod_types': [],
            'sense_mod_positions': [],
            'anti_mod_types': [],
            'anti_mod_positions': [],
            'concentrations': [],
            'pcts': []
        }
        for idx in indices:
            sample = self[idx]
            batch_data['sense_ids'].append(sample['sense_id'])
            batch_data['anti_ids'].append(sample['anti_id'])
            batch_data['sense_seqs'].append(sample['sense_seq'])
            batch_data['anti_seqs'].append(sample['anti_seq'])
            batch_data['sense_mod_types'].append(sample['sense_mod_types'])
            batch_data['sense_mod_positions'].append(sample['sense_mod_positions'])
            batch_data['anti_mod_types'].append(sample['anti_mod_types'])
            batch_data['anti_mod_positions'].append(sample['anti_mod_positions'])
            batch_data['concentrations'].append(sample['concentration'])
            batch_data['pcts'].append(sample['pct'])
        return batch_data

def collate_fn(batch):
    sense_ids = [sample['sense_id'] for sample in batch]
    anti_ids = [sample['anti_id'] for sample in batch]
    sense_seqs = [sample['sense_seq'] for sample in batch]
    anti_seqs = [sample['anti_seq'] for sample in batch]
    sense_mod_types = [sample['sense_mod_types'] for sample in batch]
    sense_mod_positions = [sample['sense_mod_positions'] for sample in batch]
    anti_mod_types = [sample['anti_mod_types'] for sample in batch]
    anti_mod_positions = [sample['anti_mod_positions'] for sample in batch]
    concentrations = torch.tensor([sample['concentration'] for sample in batch], dtype=torch.float32)
    pcts = torch.tensor([sample['pct'] for sample in batch], dtype=torch.float32)

    return {
        'sense_ids': sense_ids,
        'anti_ids': anti_ids,
        'sense_seqs': sense_seqs,
        'anti_seqs': anti_seqs,
        'sense_mod_types': sense_mod_types,
        'sense_mod_positions': sense_mod_positions,
        'anti_mod_types': anti_mod_types,
        'anti_mod_positions': anti_mod_positions,
        'concentrations': concentrations,
        'pcts': pcts
    }
