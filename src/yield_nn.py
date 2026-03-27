#!/usr/bin/env python
"""
Deep Learning for Crop Yield Prediction (Pooled Model)
======================================================
Trains a single NN across ALL regions simultaneously, using entity
embeddings for region/country (Guo & Berkhahn, 2016).

Three architectures:
  1. Pooled MLP + Entity Embeddings
  2. FT-Transformer (Gorishniy et al., 2021)
  3. 1D-CNN on monthly feature sequences

Uses cached yield_datasets/*.csv — no reprocessing.
Run from project root:  python src/yield_nn.py

Outputs:  models/nn_yield_pooled_results.pkl
"""

import os, sys, pickle, time as timer
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────
YIELD_DATA_DIR = "notebooks/yield_datasets"
MODEL_OUT_DIR  = "models"
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

YIELD_TEST_START = 2016
YIELD_TEST_END   = 2020

# ═════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═════════════════════════════════════════════════════════════

print("Loading yield datasets from cache...")
yield_datasets = {}
for lt in ['end-of-season', 'mid-season', 'quarter-season']:
    path = os.path.join(YIELD_DATA_DIR, f"{lt}.csv")
    if os.path.exists(path):
        yield_datasets[lt] = pd.read_csv(path)
        print(f"  {lt}: {yield_datasets[lt].shape}")
    else:
        print(f"  {lt}: NOT FOUND")

if 'end-of-season' not in yield_datasets:
    print("ERROR: end-of-season.csv not found. Run Section 6.2 of the notebook first.")
    sys.exit(1)

yds = yield_datasets['end-of-season']
print(f"\nDataset: {yds.shape[0]} region-years, {yds['adm_id'].nunique()} regions, "
      f"years {yds['harvest_year'].min()}-{yds['harvest_year'].max()}")

# ═════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING & METADATA
# ═════════════════════════════════════════════════════════════

print("Building production weights for fair EU-level aggregation...")
production_weights = {}
for cc in ['FR', 'DE', 'PL']:
    path = f"notebooks/cy-bench/{cc}/yield_wheat_{cc}.csv"
    if os.path.exists(path):
        raw_df = pd.read_csv(path)
        area_col = 'harvest_area' if 'harvest_area' in raw_df.columns else ('planted_area' if 'planted_area' in raw_df.columns else None)
        if area_col:
            raw_df = raw_df.sort_values(['adm_id', 'harvest_year'])
            raw_df[area_col] = raw_df.groupby('adm_id')[area_col].ffill().bfill()
        if 'production' not in raw_df.columns:
            raw_df['production'] = np.nan
        missing = raw_df['production'].isna()
        if missing.any() and area_col:
            raw_df.loc[missing, 'production'] = raw_df.loc[missing, 'yield'] * raw_df.loc[missing, area_col]
        
        for _, row in raw_df.iterrows():
            adm, yr, prod = row['adm_id'], row['harvest_year'], row['production']
            if pd.notna(prod) and prod > 0:
                if adm not in production_weights:
                    production_weights[adm] = {}
                production_weights[adm][yr] = prod

def get_production_weight(adm_id, year):
    if adm_id not in production_weights: return 0.0
    if (year - 1) in production_weights[adm_id]: return production_weights[adm_id][year - 1]
    earlier = [y for y in sorted(production_weights[adm_id].keys()) if y < year]
    if earlier: return production_weights[adm_id][earlier[-1]]
    avail = sorted(production_weights[adm_id].keys())
    return production_weights[adm_id][avail[0]] if avail else 0.0

feat_cols = [c for c in yds.columns
             if c not in ['adm_id', 'country', 'harvest_year', 'yield']]

# Encode region and country as integers for embeddings
region_enc = LabelEncoder()
country_enc = LabelEncoder()
yds['region_id'] = region_enc.fit_transform(yds['adm_id'])
yds['country_id'] = country_enc.fit_transform(yds['country'])

n_regions = yds['region_id'].nunique()
n_countries = yds['country_id'].nunique()
n_features = len(feat_cols)

print(f"Features: {n_features}, Regions: {n_regions}, Countries: {n_countries}")

# ═════════════════════════════════════════════════════════════
# 3. ARCHITECTURES
# ═════════════════════════════════════════════════════════════

# ── 3a. Pooled MLP with Entity Embeddings ─────────────────
class PooledMLP(nn.Module):
    """
    MLP with learned embeddings for region and country.
    Guo & Berkhahn (2016): entity embeddings let NNs capture
    categorical structure without one-hot explosion.
    """
    def __init__(self, n_features, n_regions, n_countries,
                 hidden_dims=(256, 128), dropout=0.3,
                 region_emb_dim=16, country_emb_dim=4):
        super().__init__()
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        self.country_emb = nn.Embedding(n_countries, country_emb_dim)

        input_dim = n_features + region_emb_dim + country_emb_dim
        layers = []
        prev = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev, hd),
                nn.LayerNorm(hd),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = hd
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, region_ids, country_ids):
        r_emb = self.region_emb(region_ids)
        c_emb = self.country_emb(country_ids)
        x = torch.cat([x_num, r_emb, c_emb], dim=1)
        return self.net(x).squeeze(-1)


# ── 3b. FT-Transformer (simplified) ──────────────────────
class FeatureTokenizer(nn.Module):
    """Each numerical feature gets its own linear projection to d_token."""
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_features, d_token))
        self.biases = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x):
        # x: (batch, n_features) -> (batch, n_features, d_token)
        return x.unsqueeze(-1) * self.weights + self.biases


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer (Gorishniy et al., 2021).
    Each feature becomes a token; self-attention captures interactions.
    """
    def __init__(self, n_features, n_regions, n_countries,
                 d_token=32, n_heads=4, n_layers=2, dropout=0.2,
                 region_emb_dim=16, country_emb_dim=4):
        super().__init__()
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        self.country_emb = nn.Embedding(n_countries, country_emb_dim)

        self.n_extra_tokens = 2  # region + country
        total_features = n_features + self.n_extra_tokens

        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.extra_proj = nn.Linear(region_emb_dim + country_emb_dim, d_token)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=d_token * 4, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x_num, region_ids, country_ids):
        batch = x_num.size(0)

        # Tokenise numerical features
        feat_tokens = self.tokenizer(x_num)  # (B, F, d)

        # Entity embedding token
        r_emb = self.region_emb(region_ids)
        c_emb = self.country_emb(country_ids)
        entity = self.extra_proj(torch.cat([r_emb, c_emb], dim=1))  # (B, d)
        entity = entity.unsqueeze(1)  # (B, 1, d)

        # CLS token
        cls = self.cls_token.expand(batch, -1, -1)

        # Concat: [CLS, entity, feat1, feat2, ...]
        tokens = torch.cat([cls, entity, feat_tokens], dim=1)

        out = self.transformer(tokens)
        cls_out = out[:, 0, :]  # CLS token output
        return self.head(cls_out).squeeze(-1)


# ── 3c. 1D-CNN on monthly sequences ──────────────────────
class TemporalCNN(nn.Module):
    """
    Treats features as a (months × variables) grid and applies
    temporal convolutions. Inspired by Khaki et al. (2020).
    """
    def __init__(self, n_features, n_regions, n_countries,
                 n_channels=64, dropout=0.3,
                 region_emb_dim=16, country_emb_dim=4):
        super().__init__()
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        self.country_emb = nn.Embedding(n_countries, country_emb_dim)

        # Treat the feature vector as 1D sequence
        self.conv1 = nn.Conv1d(1, n_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(n_channels)
        self.conv2 = nn.Conv1d(n_channels, n_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(n_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)

        head_dim = n_channels + region_emb_dim + country_emb_dim
        self.head = nn.Sequential(
            nn.Linear(head_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_num, region_ids, country_ids):
        # x_num: (B, F) -> (B, 1, F) for Conv1d
        x = x_num.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # (B, channels)

        r_emb = self.region_emb(region_ids)
        c_emb = self.country_emb(country_ids)
        x = torch.cat([x, r_emb, c_emb], dim=1)
        return self.head(x).squeeze(-1)


# ═════════════════════════════════════════════════════════════
# 4. TRAINING LOOP
# ═════════════════════════════════════════════════════════════

def train_yield_nn(model, X_train, region_train, country_train, y_train,
                   lr=1e-3, weight_decay=1e-3, epochs=300,
                   batch_size=128, patience=30, verbose=False):
    n = len(X_train)
    n_val = max(int(n * 0.2), 30)

    # Temporal split: last 20% for validation
    X_tr = torch.FloatTensor(X_train[:-n_val])
    y_tr = torch.FloatTensor(y_train[:-n_val])
    r_tr = torch.LongTensor(region_train[:-n_val])
    c_tr = torch.LongTensor(country_train[:-n_val])

    X_val = torch.FloatTensor(X_train[-n_val:])
    y_val = torch.FloatTensor(y_train[-n_val:])
    r_val = torch.LongTensor(region_train[-n_val:])
    c_val = torch.LongTensor(country_train[-n_val:])

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=10
    )
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_tr, r_tr, c_tr, y_tr),
        batch_size=batch_size, shuffle=True
    )

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, rb, cb, yb in loader:
            optimiser.zero_grad()
            loss = criterion(model(xb, rb, cb), yb)
            loss.backward()
            optimiser.step()

        val_loader = DataLoader(
            TensorDataset(X_val, r_val, c_val, y_val),
            batch_size=batch_size, shuffle=False
        )
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, rb, cb, yb in val_loader:
                loss = criterion(model(xb, rb, cb), yb)
                val_loss += loss.item() * len(yb)
            val_loss /= len(y_val)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            if verbose:
                print(f'  Early stop epoch {epoch+1}, val_loss={best_val_loss:.6f}')
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ═════════════════════════════════════════════════════════════
# 5. WALK-FORWARD EVALUATION
# ═════════════════════════════════════════════════════════════

def walk_forward_yield(yds, model_cls, model_kwargs,
                       train_kwargs=None,
                       test_start=YIELD_TEST_START,
                       test_end=YIELD_TEST_END):
    """
    Walk-forward pooled yield prediction.
    Train on ALL regions for years < T, predict ALL regions for year T.
    """
    if train_kwargs is None:
        train_kwargs = {}

    all_records = []

    for ty in range(test_start, test_end + 1):
        train = yds[yds['harvest_year'] < ty]
        test = yds[yds['harvest_year'] == ty]

        if len(train) < 50 or len(test) < 5:
            continue

        sc = StandardScaler()
        X_tr = sc.fit_transform(train[feat_cols].values)
        X_te = sc.transform(test[feat_cols].values)
        
        y_sc = StandardScaler()
        y_tr_scaled = y_sc.fit_transform(train['yield'].values.reshape(-1, 1)).flatten()
        
        y_te = test['yield'].values
        r_tr = train['region_id'].values
        r_te = test['region_id'].values
        c_tr = train['country_id'].values
        c_te = test['country_id'].values

        model = model_cls(n_features, n_regions, n_countries, **model_kwargs)
        model = train_yield_nn(model, X_tr, r_tr, c_tr, y_tr_scaled, **train_kwargs)

        with torch.no_grad():
            y_pred_scaled = model(
                torch.FloatTensor(X_te),
                torch.LongTensor(r_te),
                torch.LongTensor(c_te)
            ).numpy()
            
        y_pred = y_sc.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        for idx, (_, row) in enumerate(test.iterrows()):
            w = get_production_weight(row['adm_id'], ty)
            all_records.append({
                'harvest_year': ty,
                'adm_id': row['adm_id'],
                'country': row['country'],
                'y_true': y_te[idx],
                'y_pred': y_pred[idx],
                'weight': w,
            })

    if not all_records:
        return None

    rdf = pd.DataFrame(all_records)

    # Per-region metrics
    per_r2 = r2_score(rdf['y_true'], rdf['y_pred'])
    per_rmse = np.sqrt(mean_squared_error(rdf['y_true'], rdf['y_pred']))
    per_dir_acc = np.mean(np.sign(rdf['y_true']) == np.sign(rdf['y_pred']))

    # Production-weighted aggregation to EU level per year
    eu_records = []
    for yr in rdf['harvest_year'].unique():
        yr_df = rdf[rdf['harvest_year'] == yr]
        total_w = yr_df['weight'].sum()
        if total_w > 0:
            agg_true = np.average(yr_df['y_true'], weights=yr_df['weight'])
            agg_pred = np.average(yr_df['y_pred'], weights=yr_df['weight'])
        else:
            agg_true = yr_df['y_true'].mean()
            agg_pred = yr_df['y_pred'].mean()
            
        eu_records.append({
            'harvest_year': yr,
            'y_true': agg_true,
            'y_pred': agg_pred,
        })
    eu_df = pd.DataFrame(eu_records)
    eu_r2 = r2_score(eu_df['y_true'], eu_df['y_pred']) if len(eu_df) > 1 else np.nan
    eu_rmse = np.sqrt(mean_squared_error(eu_df['y_true'], eu_df['y_pred']))
    eu_dir_acc = np.mean(np.sign(eu_df['y_true']) == np.sign(eu_df['y_pred']))

    return {
        'per_region': {'r2': per_r2, 'rmse': per_rmse, 'dir_acc': per_dir_acc},
        'eu_agg': {'r2': eu_r2, 'rmse': eu_rmse, 'dir_acc': eu_dir_acc},
        'eu_df': eu_df,
        'all_preds': rdf,
    }


# ═════════════════════════════════════════════════════════════
# 6. RUN ALL ARCHITECTURES
# ═════════════════════════════════════════════════════════════

CONFIGS = {
    # ── Pooled MLP variants ───────────────────────────────
    'MLP-Small (128-64)': {
        'cls': PooledMLP,
        'kwargs': {'hidden_dims': (128, 64), 'dropout': 0.3},
        'train': {'lr': 1e-3, 'weight_decay': 1e-3, 'epochs': 300, 'patience': 30},
    },
    'MLP-Medium (256-128)': {
        'cls': PooledMLP,
        'kwargs': {'hidden_dims': (256, 128), 'dropout': 0.3},
        'train': {'lr': 1e-3, 'weight_decay': 1e-3, 'epochs': 300, 'patience': 30},
    },
    'MLP-Deep (256-128-64)': {
        'cls': PooledMLP,
        'kwargs': {'hidden_dims': (256, 128, 64), 'dropout': 0.4},
        'train': {'lr': 5e-4, 'weight_decay': 1e-3, 'epochs': 300, 'patience': 30},
    },
    'MLP-Heavy-Reg (256-128)': {
        'cls': PooledMLP,
        'kwargs': {'hidden_dims': (256, 128), 'dropout': 0.5},
        'train': {'lr': 1e-3, 'weight_decay': 5e-2, 'epochs': 300, 'patience': 30},
    },
    # ── FT-Transformer ────────────────────────────────────
    'FT-Transformer (d=32, L=2)': {
        'cls': FTTransformer,
        'kwargs': {'d_token': 32, 'n_heads': 4, 'n_layers': 2, 'dropout': 0.2},
        'train': {'lr': 1e-4, 'weight_decay': 1e-3, 'epochs': 200, 'patience': 25},
    },
    'FT-Transformer (d=64, L=3)': {
        'cls': FTTransformer,
        'kwargs': {'d_token': 64, 'n_heads': 4, 'n_layers': 3, 'dropout': 0.3},
        'train': {'lr': 1e-4, 'weight_decay': 1e-3, 'epochs': 200, 'patience': 25},
    },
    # ── 1D-CNN ────────────────────────────────────────────
    '1D-CNN (64ch)': {
        'cls': TemporalCNN,
        'kwargs': {'n_channels': 64, 'dropout': 0.3},
        'train': {'lr': 1e-3, 'weight_decay': 1e-3, 'epochs': 300, 'patience': 30},
    },
    '1D-CNN (128ch)': {
        'cls': TemporalCNN,
        'kwargs': {'n_channels': 128, 'dropout': 0.3},
        'train': {'lr': 5e-4, 'weight_decay': 1e-3, 'epochs': 300, 'patience': 30},
    },
}


if __name__ == '__main__':
    print("\n" + "=" * 74)
    print("DEEP LEARNING YIELD PREDICTION: Pooled Model (walk-forward 2016-2020)")
    print("=" * 74)
    print(f"Regions: {n_regions}  |  Features: {n_features}  |  "
          f"Countries: {n_countries}  |  Samples: {len(yds)}")

    all_results = {}

    for name, cfg in CONFIGS.items():
        t0 = timer.time()
        print(f"\n{'─'*74}")
        print(f"  {name}")
        print(f"{'─'*74}")

        res = walk_forward_yield(
            yds, cfg['cls'], cfg['kwargs'],
            train_kwargs=cfg['train']
        )

        elapsed = timer.time() - t0

        if res:
            all_results[name] = res
            rr = res['per_region']
            eu = res['eu_agg']
            print(f"  Reg R²={rr['r2']:+.4f}  Reg RMSE={rr['rmse']:.4f}  Reg DirAcc={rr['dir_acc']:.4f}  "
                  f"EU R²={eu['r2']:+.4f}  EU RMSE={eu['rmse']:.4f}  EU DirAcc={eu['dir_acc']:.4f}  ({elapsed:.0f}s)")
        else:
            print(f"  FAILED (insufficient data)")

    # ── Summary Table ─────────────────────────────────────
    print(f"\n{'='*74}")
    print("SUMMARY: Deep Learning Yield Prediction (Test 2016-2020)")
    print(f"{'='*74}")
    print(f"\n{'Model':30s} {'Reg R²':>8s} {'Reg RMSE':>10s} "
          f"{'Reg DirAcc':>10s} {'EU R²':>7s} {'EU RMSE':>9s} {'EU DirAcc':>10s}")
    print('-' * 91)

    for name, res in all_results.items():
        rr = res['per_region']
        eu = res['eu_agg']
        print(f"{name:30s} {rr['r2']:+8.4f} {rr['rmse']:10.4f} "
              f"{rr['dir_acc']:10.4f} {eu['r2']:+7.4f} {eu['rmse']:9.4f} {eu['dir_acc']:10.4f}")

    # ── Save ──────────────────────────────────────────────
    out_path = os.path.join(MODEL_OUT_DIR, 'nn_yield_pooled_results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\n✓ Saved results → {out_path}")

    # ── Lead-time comparison (if mid/quarter available) ───
    for lt in ['mid-season', 'quarter-season']:
        if lt not in yield_datasets:
            continue
        lt_yds = yield_datasets[lt].copy()
        lt_yds['region_id'] = region_enc.transform(lt_yds['adm_id'])
        lt_yds['country_id'] = country_enc.transform(lt_yds['country'])

        # Use best MLP config
        best_name = min(all_results.keys(),
                        key=lambda k: all_results[k]['per_region']['nrmse'])
        cfg = CONFIGS[best_name]

        print(f"\n── {lt} (using {best_name}) ──")
        res = walk_forward_yield(lt_yds, cfg['cls'], cfg['kwargs'],
                                 train_kwargs=cfg['train'])
        if res:
            rr = res['per_region']
            eu = res['eu_agg']
            print(f"  Reg R²={rr['r2']:+.4f}  RMSE={rr['rmse']:.4f}  DirAcc={rr['dir_acc']:.4f}  "
                  f"EU R²={eu['r2']:+.4f}  RMSE={eu['rmse']:.4f}  DirAcc={eu['dir_acc']:.4f}")

    print("\n✓ All evaluations complete!")
