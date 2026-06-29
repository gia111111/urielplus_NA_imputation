from __future__ import annotations
import argparse
import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')
_THIS = Path(__file__).resolve()
_REPO_SRC = _THIS.parent.parent / 'src'
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
from urielplus_impute.data import load_dataset
from urielplus_impute.metrics import compute_binary_imputation_metrics, compute_stratified_metrics
from urielplus_impute.results import write_result_tables
from urielplus_impute.softimpute import SoftImpute as _ProjectSoftImpute
from urielplus_impute.split_io import load_split, load_split_manifest_rows
DEFAULT_EPOCHS = 40
DEFAULT_BATCH = 256
DEFAULT_LR = 0.001
DEFAULT_W_POS = 2.33
DEFAULT_W_NEG = 1.0

@dataclass
class Context:
    family_arr: np.ndarray
    macro_arr: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    fam_sizes: np.ndarray
    lang_type: np.ndarray
    side_features: np.ndarray
    val_rows: np.ndarray = field(default_factory=lambda : np.array([], dtype=np.int64))
    val_cols: np.ndarray = field(default_factory=lambda : np.array([], dtype=np.int64))
    val_true: np.ndarray = field(default_factory=lambda : np.array([], dtype=np.float32))
    regime_name: str = ''

def build_context(dataset, regime_name: str='') -> Context:
    languages: pd.DataFrame = dataset.languages
    family_arr = languages['family_id'].to_numpy().astype(str)
    macro_arr = languages['macroarea'].to_numpy().astype(str)
    fam_sizes = languages['family_size'].to_numpy().astype(np.int64)
    lat_raw = pd.to_numeric(languages['latitude'], errors='coerce').to_numpy(dtype=float)
    lon_raw = pd.to_numeric(languages['longitude'], errors='coerce').to_numpy(dtype=float)
    lat_med = float(np.nanmedian(lat_raw)) if np.isfinite(np.nanmedian(lat_raw)) else 0.0
    lon_med = float(np.nanmedian(lon_raw)) if np.isfinite(np.nanmedian(lon_raw)) else 0.0
    lat = np.where(np.isnan(lat_raw), lat_med, lat_raw)
    lon = np.where(np.isnan(lon_raw), lon_med, lon_raw)
    fsg = languages['family_size_group'].to_numpy().astype(str)
    fsg_to_lang_type = {'isolate_or_unknown': 'isolate', 'small_family_2_4': 'small_family', 'medium_family_5_19': 'large_family', 'large_family_20_plus': 'large_family'}
    lang_type = np.array([fsg_to_lang_type.get(g, 'large_family') for g in fsg])
    side_features = _build_side_features(family_arr, macro_arr, fam_sizes, lat, lon)
    return Context(family_arr=family_arr, macro_arr=macro_arr, lat=lat, lon=lon, fam_sizes=fam_sizes, lang_type=lang_type, side_features=side_features, regime_name=regime_name)

def _build_side_features(family_arr: np.ndarray, macro_arr: np.ndarray, fam_sizes: np.ndarray, lat_f: np.ndarray, lon_f: np.ndarray, top_k_families: int=50) -> np.ndarray:
    n_lang = len(family_arr)
    top_fams = pd.Series(family_arr).value_counts().head(top_k_families).index.tolist()
    fam_oh = np.zeros((n_lang, top_k_families + 1), dtype=np.float32)
    for (i, fam) in enumerate(family_arr):
        if fam in top_fams:
            fam_oh[i, top_fams.index(fam)] = 1.0
        else:
            fam_oh[i, top_k_families] = 1.0
    unique_macros = sorted(set(macro_arr))
    mac_oh = np.zeros((n_lang, len(unique_macros)), dtype=np.float32)
    for (i, mac) in enumerate(macro_arr):
        mac_oh[i, unique_macros.index(mac)] = 1.0
    log_size = (np.log(fam_sizes.astype(float) + 1) / np.log(fam_sizes.max() + 1)).astype(np.float32)
    lat_n = ((lat_f - lat_f.min()) / (lat_f.max() - lat_f.min() + 1e-09)).astype(np.float32)
    lon_n = ((lon_f - lon_f.min()) / (lon_f.max() - lon_f.min() + 1e-09)).astype(np.float32)
    return np.concatenate([fam_oh, mac_oh, log_size.reshape(-1, 1), lat_n.reshape(-1, 1), lon_n.reshape(-1, 1)], axis=1)

def _softimpute(mat: np.ndarray, max_iters: int=100) -> np.ndarray:
    arr = mat.astype(np.float64)
    try:
        return _ProjectSoftImpute(max_iters=max_iters, verbose=False).fit(pd.DataFrame(arr)).imputed_matrix_
    except np.linalg.LinAlgError:
        return _softimpute_scipy_fallback(arr, max_iters=max_iters)

def _softimpute_scipy_fallback(mat: np.ndarray, max_iters: int=100, shrinkage: float=1.0, tol: float=0.0001) -> np.ndarray:
    from scipy.linalg import svd as scipy_svd
    observed = ~np.isnan(mat)
    missing = ~observed
    filled = np.nan_to_num(mat, nan=0.0)
    counts = observed.sum(axis=0).astype(float)
    sums = filled.sum(axis=0)
    overall = float(sums.sum() / max(counts.sum(), 1.0))
    feature_means = np.divide(sums, counts, out=np.full(mat.shape[1], overall), where=counts > 0)
    current = np.clip(np.where(observed, mat, feature_means[None, :]), 0.0, 1.0)
    if not np.any(missing):
        return current
    previous_missing = current[missing].copy()
    for _ in range(max_iters):
        (u, s, vt) = scipy_svd(current, full_matrices=False, lapack_driver='gesvd')
        shrunk = np.maximum(s - shrinkage, 0.0)
        keep = np.flatnonzero(shrunk > 0.0)
        if len(keep):
            reconstructed = u[:, keep] * shrunk[keep] @ vt[keep, :]
        else:
            reconstructed = np.tile(feature_means[None, :], (mat.shape[0], 1))
        reconstructed = np.clip(reconstructed, 0.0, 1.0)
        current = np.where(observed, mat, reconstructed)
        current_missing = current[missing]
        denom = max(float(np.linalg.norm(previous_missing)), 1e-12)
        rel = float(np.linalg.norm(current_missing - previous_missing) / denom)
        previous_missing = current_missing.copy()
        if rel < tol:
            break
    return np.clip(current, 0.0, 1.0)

def compute_loc_glob_fill(X_train: np.ndarray, family_arr: np.ndarray, min_family_obs: int=10, blend_const: float=50.0) -> np.ndarray:
    X_global = _softimpute(X_train)
    X_imp = X_global.copy()
    (uf, fc) = np.unique(family_arr, return_counts=True)
    for fam in uf[fc >= 10]:
        members = np.where(family_arr == fam)[0]
        if (~np.isnan(X_train[members])).sum() < min_family_obs:
            continue
        n_mem = len(members)
        X_fam = _softimpute(X_train[members])
        a_fam = min(0.7, n_mem / (n_mem + blend_const))
        X_imp[members] = a_fam * X_fam + (1 - a_fam) * X_global[members]
    return np.clip(X_imp, 0.0, 1.0)

class ResBlock(nn.Module):

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(x + self.net(x))

class NN_side(nn.Module):

    def __init__(self, d_feat: int, d_side: int, h: int=512):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_feat + d_side, h), nn.GELU())
        self.r1 = ResBlock(h)
        self.r2 = ResBlock(h)
        self.r3 = ResBlock(h)
        self.dec = nn.Sequential(nn.Linear(h, d_feat), nn.Sigmoid())

    def forward(self, x, side, **kwargs):
        z = self.enc(torch.cat([x, side], dim=-1))
        return self.dec(self.r3(self.r2(self.r1(z))))

def weighted_mse(pred, target, mask, w_pos=DEFAULT_W_POS, w_neg=DEFAULT_W_NEG):
    w = target * w_pos + (1 - target) * w_neg
    num = ((pred - target) ** 2 * mask * w).sum()
    den = (mask * w).sum() + 1e-08
    return num / den

def _make_dorcor_corruption(xb, ob, donor):
    cmask = (torch.rand_like(xb) < 0.2) & ob.bool()
    xc = xb.clone()
    xc[cmask] = donor[cmask]
    extra = torch.rand_like(xb) < 0.15
    tgt = ob * (cmask.float() + extra.float()).clamp(0, 1)
    x_in = xc * (1 - extra.float()) + 0.5 * extra.float()
    return (x_in, tgt)

def train_dorcor(X_obs_t, side_t, obs_t, model, seed, n_lang, epochs: int=DEFAULT_EPOCHS, batch: int=DEFAULT_BATCH, lr: float=DEFAULT_LR):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n_lang)
        for i in range(0, n_lang, batch):
            idx = perm[i:i + batch]
            bs = len(idx)
            (xb, ob) = (X_obs_t[idx], obs_t[idx])
            side_b = side_t[idx]
            donor = X_obs_t[torch.randint(0, n_lang, (bs,))]
            (x_in, tgt) = _make_dorcor_corruption(xb, ob, donor)
            pred = model(x_in, side_b)
            loss = weighted_mse(pred, xb, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model

def train_orig(X_obs_t, side_t, obs_t, model, seed, n_lang, epochs: int=DEFAULT_EPOCHS, batch: int=DEFAULT_BATCH, lr: float=DEFAULT_LR):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n_lang)
        for i in range(0, n_lang, batch):
            idx = perm[i:i + batch]
            (xb, ob) = (X_obs_t[idx], obs_t[idx])
            side_b = side_t[idx]
            extra = torch.rand_like(xb) < 0.3
            tgt = ob * extra.float()
            pred = model(xb, side_b)
            loss = weighted_mse(pred, xb, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model

def predict_full(model, X_obs_t, side_t, X_train: np.ndarray, batch: int=DEFAULT_BATCH) -> np.ndarray:
    n_lang = X_obs_t.shape[0]
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, n_lang, batch):
            idx_t = torch.arange(i, min(i + batch, n_lang))
            preds.append(model(X_obs_t[idx_t], side_t[idx_t]).cpu().numpy())
    P = np.vstack(preds)
    return np.where(np.isnan(X_train), P, X_train).astype(np.float64)

def tune_alpha_3strata(X_orig: np.ndarray, X_dorcor: np.ndarray, val_rows: np.ndarray, val_cols: np.ndarray, val_true: np.ndarray, lang_type: np.ndarray) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for g in ('isolate', 'small_family', 'large_family'):
        idx = lang_type[val_rows] == g
        if idx.sum() == 0:
            weights[g] = 0.5
            continue
        a = np.clip(X_orig[val_rows[idx], val_cols[idx]], 0, 1)
        b = np.clip(X_dorcor[val_rows[idx], val_cols[idx]], 0, 1)
        tv = val_true[idx].astype(float)
        (best_a, best_r) = (0.0, float('inf'))
        for alpha in np.round(np.arange(0.0, 1.01, 0.02), 2):
            r = float(np.sqrt(np.mean((alpha * a + (1 - alpha) * b - tv) ** 2)))
            if r < best_r:
                (best_r, best_a) = (r, float(alpha))
        weights[g] = best_a
    return weights

def apply_ensemble(X_orig: np.ndarray, X_dorcor: np.ndarray, alpha_per_group: Dict[str, float], lang_type: np.ndarray) -> np.ndarray:
    a_vec = np.array([alpha_per_group[lt] for lt in lang_type])[:, None]
    return a_vec * X_orig + (1 - a_vec) * X_dorcor

class deep_model:
    name = 'deep_model'

    def __init__(self, *, seed: int=42, context: Optional[Context]=None, val_cells: Optional[pd.DataFrame]=None, regime: str='', epochs: int=DEFAULT_EPOCHS, batch: int=DEFAULT_BATCH, lr: float=DEFAULT_LR):
        if context is None:
            raise ValueError('deep_model requires a Context (build via build_context(dataset)).')
        self.seed = int(seed)
        self._context = context
        self._val_cells = val_cells
        self._regime = regime
        self.epochs = int(epochs)
        self.batch = int(batch)
        self.lr = float(lr)
        self._predicted_matrix: Optional[np.ndarray] = None

    def fit(self, X_train: pd.DataFrame) -> 'deep_model':
        ctx = self._context
        if X_train.shape[0] != len(ctx.family_arr):
            raise ValueError(f'X_train rows ({X_train.shape[0]}) ≠ Context langs ({len(ctx.family_arr)}). Build Context from the same dataset that produced these splits.')
        X_train_np = X_train.to_numpy(dtype=np.float64)
        if self._val_cells is not None and len(self._val_cells) > 0:
            ctx.val_rows = self._val_cells['row_idx'].to_numpy(dtype=np.int64)
            ctx.val_cols = self._val_cells['col_idx'].to_numpy(dtype=np.int64)
            ctx.val_true = self._val_cells['true_value'].to_numpy(dtype=np.float32)
        ctx.regime_name = self._regime
        (n_lang, n_feat) = X_train_np.shape
        X_imp = compute_loc_glob_fill(X_train_np, ctx.family_arr)
        x_obs = np.where(np.isnan(X_train_np), np.clip(X_imp, 0, 1), X_train_np).astype(np.float32)
        side = ctx.side_features.astype(np.float32)
        X_obs_t = torch.tensor(x_obs)
        side_t = torch.tensor(side)
        obs_t = torch.tensor((~np.isnan(X_train_np)).astype(np.float32))
        m_dorcor = NN_side(n_feat, side.shape[1])
        m_orig = NN_side(n_feat, side.shape[1])
        m_dorcor = train_dorcor(X_obs_t, side_t, obs_t, m_dorcor, self.seed, n_lang, epochs=self.epochs, batch=self.batch, lr=self.lr)
        m_orig = train_orig(X_obs_t, side_t, obs_t, m_orig, self.seed, n_lang, epochs=self.epochs, batch=self.batch, lr=self.lr)
        X_pred_dorcor = predict_full(m_dorcor, X_obs_t, side_t, X_train_np, self.batch)
        X_pred_orig = predict_full(m_orig, X_obs_t, side_t, X_train_np, self.batch)
        if len(ctx.val_rows) > 0:
            alpha = tune_alpha_3strata(X_pred_orig, X_pred_dorcor, ctx.val_rows, ctx.val_cols, ctx.val_true, ctx.lang_type)
        else:
            alpha = {'isolate': 0.5, 'small_family': 0.5, 'large_family': 0.5}
        self._predicted_matrix = np.clip(apply_ensemble(X_pred_orig, X_pred_dorcor, alpha, ctx.lang_type), 0.0, 1.0)
        self._alpha = alpha
        return self

    def predict_cells(self, cells: pd.DataFrame) -> np.ndarray:
        if self._predicted_matrix is None:
            raise RuntimeError('predict_cells() called before fit()')
        rows = cells['row_idx'].astype(int).to_numpy()
        cols = cells['col_idx'].astype(int).to_numpy()
        return np.clip(self._predicted_matrix[rows, cols], 0.0, 1.0)
DEFAULT_SPLIT_MANIFEST = 'imputation_exp/runs/splits/manifests/split_manifest.csv'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the deep_model deep imputer on saved URIEL+ splits.')
    parser.add_argument('--typological', default='urielplus_analysis/typological_data.csv')
    parser.add_argument('--languages', default='urielplus_analysis/languages.csv')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--split-manifest', default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument('--regimes', nargs='+', default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    parser.add_argument('--index-col', default=None)
    parser.add_argument('--drop-empty-languages', action='store_true')
    parser.add_argument('--keep-special-languages', action='store_true')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--save-predictions', action='store_true', help='Persist per-cell predictions to <outdir>/predictions/')
    return parser.parse_args()

def _save_predictions(outdir: Path, regime: str, seed: int, split_name: str, cells: pd.DataFrame, scores: np.ndarray) -> None:
    pred_dir = outdir / 'predictions' / regime
    pred_dir.mkdir(parents=True, exist_ok=True)
    df = cells.copy()
    df['y_score'] = np.clip(scores, 0.0, 1.0)
    df['y_pred'] = (df['y_score'] >= 0.5).astype(int)
    df.to_csv(pred_dir / f'seed_{seed}_{split_name}.csv', index=False)

def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    metrics_dir = outdir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    print(f'[info] typological   : {args.typological}')
    print(f'[info] languages     : {args.languages}')
    print(f'[info] split manifest: {args.split_manifest}')
    print(f'[info] outdir        : {outdir}')
    print(f'[info] epochs={args.epochs}  batch={args.batch}  lr={args.lr}')
    t0 = time.time()
    dataset = load_dataset(args.typological, args.languages, index_col=args.index_col, drop_empty_languages=args.drop_empty_languages, filter_special_families=not args.keep_special_languages)
    print(f'[load] dataset loaded in {time.time() - t0:.1f}s · X={dataset.X.shape}')
    shared_ctx = build_context(dataset, regime_name='')
    print(f'[ctx ] side_features dim = {shared_ctx.side_features.shape[1]}')
    split_rows = load_split_manifest_rows(Path(args.split_manifest), regimes=args.regimes, seeds=args.seeds)
    if not split_rows:
        raise SystemExit('No splits matched the regime/seed filters.')
    print(f'[plan] {len(split_rows)} fits\n')
    manifest_out = outdir / 'manifests'
    manifest_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**row, 'split_path': str(row['split_path']), 'metadata_path': str(row['metadata_path'])} for row in split_rows]).to_csv(manifest_out / 'deep_model_split_manifest.csv', index=False)
    seed_metric_rows: list[dict] = []
    stratified_rows: list[pd.DataFrame] = []
    overall_t0 = time.time()
    for (split_idx, row) in enumerate(split_rows, start=1):
        loaded = load_split(row, dataset.X, dataset.feature_types)
        print(f'\n[split {split_idx}/{len(split_rows)}] regime={loaded.regime}  seed={loaded.seed}  train_visible={int(loaded.masks.train_visible_mask.sum()):,}  val={len(loaded.val_cells):,}  test={len(loaded.test_cells):,}')
        t_fit = time.time()
        try:
            model = deep_model(seed=loaded.seed, context=shared_ctx, val_cells=loaded.val_cells, regime=loaded.regime, epochs=args.epochs, batch=args.batch, lr=args.lr)
            model.fit(loaded.train_matrix)
            val_scores = model.predict_cells(loaded.val_cells)
            test_scores = model.predict_cells(loaded.test_cells)
        except Exception as e:
            elapsed = time.time() - t_fit
            print(f'  FAILED in {elapsed:.1f}s — {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            continue
        elapsed = time.time() - t_fit
        val_m = compute_binary_imputation_metrics(loaded.val_cells['true_value'], val_scores, threshold=0.5)
        test_m = compute_binary_imputation_metrics(loaded.test_cells['true_value'], test_scores, threshold=0.5)
        print(f"  done in {elapsed:.1f}s · val RMSE={val_m['rmse']:.4f}  test RMSE={test_m['rmse']:.4f}  F1={test_m['macro_f1']:.4f}")
        for (split_name, cells, scores, metrics) in (('val', loaded.val_cells, val_scores, val_m), ('test', loaded.test_cells, test_scores, test_m)):
            seed_metric_rows.append({'model': 'deep_model', 'variant': 'deep_model', 'regime': loaded.regime, 'seed': loaded.seed, 'split': split_name, 'n': int(len(cells)), **metrics, 'threshold': 0.5, 'elapsed_seconds': elapsed, 'split_path': str(loaded.split_path), 'metadata_path': str(loaded.metadata_path)})
            strat = compute_stratified_metrics(cells['true_value'], scores, cells['feature_type'], threshold=0.5)
            for (k, v) in {'model': 'deep_model', 'variant': 'deep_model', 'regime': loaded.regime, 'seed': loaded.seed, 'split': split_name}.items():
                strat[k] = v
            stratified_rows.append(strat)
            if args.save_predictions:
                _save_predictions(outdir, loaded.regime, loaded.seed, split_name, cells, scores)
        pd.DataFrame(seed_metric_rows).to_csv(metrics_dir / '_partial_seed_metrics.csv', index=False)
    if not seed_metric_rows:
        print('\nNo successful fits — nothing to write.')
        return
    outputs = write_result_tables(pd.DataFrame(seed_metric_rows), pd.concat(stratified_rows, ignore_index=True) if stratified_rows else pd.DataFrame(), metrics_dir)
    for (name, path) in outputs.items():
        print(f'[wrote] {name}: {path}')
    print(f'\nALL DONE · total wall time: {(time.time() - overall_t0) / 60:.1f} min · {len(seed_metric_rows) // 2} successful fits')
if __name__ == '__main__':
    main()
