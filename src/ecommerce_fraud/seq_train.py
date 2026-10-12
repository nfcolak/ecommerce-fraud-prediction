"""Training for the sequence model: masked-field pretraining, fine-tuning, scoring, embeddings.

Smoke: PYTHONPATH=src python -m ecommerce_fraud.seq_train --smoke-synthetic
       PYTHONPATH=src python -m ecommerce_fraud.seq_train --smoke-real   (data/processed/seq/smoke.npz)
Full (needs approval, not run in the SSL-prep phase):
    python -m ecommerce_fraud.seq_train --arrays <npz> --mode pretrain|finetune|ssl_finetune \\
        --name <model> --window valid|fold:S-E --es-days 14 --seed <int> \\
        [--save-embed (ssl_finetune only)] [--models-dir DIR] [--results-dir DIR]

Inputs never contain y. Windows (ADR-002/003): the test split is never scored, fine-tuned on, used for
early stopping or pretraining (hard guard).
  valid     : fine-tune = data.time_split train rows minus the early-stopping slice (last es-days days of
              train); scored = time_split valid rows; fit_end_dt = last train dt (must equal arrays meta).
  fold:S-E  : train days 1..S-8, early-stopping = the last 14 of them, scored = labeled rows of days S..E;
              fit_end_dt = end of day S-8.
Pretraining uses every array row (labeled or not) with dt <= fit_end_dt.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from ecommerce_fraud.seq_model import FraudHead, MaskedFieldHead, SeqClassifier, SeqEncoder

DAY = 86400
MASK_FRAC = 0.15


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def device_label(device: str) -> str:
    return f"{device} ({torch.cuda.get_device_name()})" if device == "cuda" else device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- batches

def build_batch(arrays, rows, device="cpu") -> dict:
    """Gather the 31 history rows + the current row (last) for each row index. y is never touched."""
    rows = np.asarray(rows, dtype=np.int64)
    idx = np.concatenate([arrays.hist[rows].astype(np.int64), rows[:, None]], axis=1)  # [B, T]
    pad = idx < 0
    safe = np.where(pad, 0, idx)
    cat = arrays.cat[safe].astype(np.int64)
    num = arrays.num[safe].astype(np.float32)
    miss = arrays.num_missing[safe].astype(np.float32)
    cat[pad] = 0
    num[pad] = 0.0
    miss[pad] = 1.0
    gap = np.log1p(np.maximum(arrays.dt[rows][:, None] - arrays.dt[safe], 0) / 3600.0).astype(np.float32)
    gap[pad] = 0.0
    gap[:, -1] = 0.0
    out = {"cat": cat, "num": num, "miss": miss, "gap": gap, "pad": pad}
    return {k: torch.from_numpy(v).to(device) for k, v in out.items()}


def make_model(arrays, d_model=64, n_heads=4, n_layers=2, dropout=0.1) -> SeqClassifier:
    enc = SeqEncoder(arrays.num.shape[1], arrays.cat_cardinality, d_model, n_heads, n_layers, dropout)
    return SeqClassifier(enc, FraudHead(d_model, d_model, dropout))


def _cosine(opt, total_steps):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))


# --------------------------------------------------------------------------- pretraining

def mask_batch(batch):
    """Mask 15% of the fields of every real token. Returns masked batch + field masks."""
    cat, num, miss, pad = batch["cat"], batch["num"], batch["miss"], batch["pad"]
    K, Fn = cat.shape[-1], num.shape[-1]
    r = torch.rand(cat.shape[0], cat.shape[1], K + Fn, device=cat.device)
    m = (r < MASK_FRAC) & (~pad).unsqueeze(-1)
    mc, mn = m[..., :K], m[..., K:]
    masked = dict(batch)
    masked["cat"] = cat.masked_fill(mc, 0)
    masked["num"] = num.masked_fill(mn, 0.0)
    masked["miss"] = miss.masked_fill(mn, 1.0)
    return masked, mc, mn


def masked_field_loss(encoder, head, batch):
    masked, mc, mn = mask_batch(batch)
    h = encoder.forward_tokens(masked)
    cat_loss = h.new_zeros(())
    for k in range(batch["cat"].shape[-1]):
        sel = mc[..., k]
        if bool(sel.any()):
            cat_loss = cat_loss + F.cross_entropy(head.cat_logits(k, h[sel]), batch["cat"][..., k][sel])
    tm = (mn & (batch["miss"] == 0)).float()  # masked and originally observed
    sq = (head.num_pred(h) - batch["num"]) ** 2 * tm
    num_loss = (sq.sum(dim=(0, 1)) / tm.sum(dim=(0, 1)).clamp(min=1.0)).sum()
    return cat_loss + num_loss, cat_loss.detach(), num_loss.detach()


def pretrain(arrays, row_idx, epochs, seed, device=None, batch_size=1024, lr=1e-3, **model_kw):
    """Masked-field modeling. Returns (encoder state dict on cpu, loss log)."""
    device = device or pick_device()
    set_seed(seed)
    enc = make_model(arrays, **model_kw).encoder.to(device)
    head = MaskedFieldHead(arrays.num.shape[1], arrays.cat_cardinality, enc.d_model).to(device)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    row_idx = np.asarray(row_idx, dtype=np.int64)
    steps_per_epoch = int(np.ceil(len(row_idx) / batch_size))
    sched = _cosine(opt, epochs * steps_per_epoch)
    rng = np.random.RandomState(seed)
    log = []
    enc.train()
    head.train()
    for ep in range(epochs):
        perm = rng.permutation(row_idx)
        tot = cat_t = num_t = 0.0
        t0 = time.time()
        for s in range(steps_per_epoch):
            batch = build_batch(arrays, perm[s * batch_size:(s + 1) * batch_size], device)
            loss, cl, nl = masked_field_loss(enc, head, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            cat_t += float(cl)
            num_t += float(nl)
        rec = {"epoch": ep + 1, "loss": tot / steps_per_epoch, "cat_loss": cat_t / steps_per_epoch,
               "num_loss": num_t / steps_per_epoch, "seconds": time.time() - t0}
        log.append(rec)
        print("pretrain", json.dumps(rec), flush=True)
    return {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}, log


# --------------------------------------------------------------------------- fine-tuning / scoring

def _model_device(model):
    return next(model.parameters()).device


@torch.no_grad()
def score(model, arrays, idx, batch_size=2048, device=None) -> np.ndarray:
    """Fraud probability per row of idx."""
    device = device or _model_device(model)
    model.to(device).eval()
    idx = np.asarray(idx, dtype=np.int64)
    out = [torch.sigmoid(model(build_batch(arrays, idx[s:s + batch_size], device))).cpu().numpy()
           for s in range(0, len(idx), batch_size)]
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


@torch.no_grad()
def embed(model, arrays, idx, batch_size=2048, device=None) -> np.ndarray:
    """Frozen current-token embeddings [len(idx), d_model] (model: SeqClassifier or SeqEncoder)."""
    device = device or _model_device(model)
    model.to(device).eval()
    idx = np.asarray(idx, dtype=np.int64)
    out = [model.embed(build_batch(arrays, idx[s:s + batch_size], device)).cpu().numpy()
           for s in range(0, len(idx), batch_size)]
    return np.concatenate(out).astype(np.float32)


def finetune(arrays, train_idx, es_idx, epochs, seed, device=None, init_state=None, pos_weight=28.0,
             batch_size=1024, lr=None, patience=3, **model_kw):
    """BCE with pos_weight, early stopping on PR-AUC of es_idx. Returns (best model, history).
    init_state: encoder state dict from pretrain(). lr default 1e-3 from scratch, 3e-4 when initialised."""
    device = device or pick_device()
    set_seed(seed)
    model = make_model(arrays, **model_kw)
    if init_state is not None:
        model.encoder.load_state_dict(init_state)
    lr = lr if lr is not None else (3e-4 if init_state is not None else 1e-3)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pw = torch.tensor(float(pos_weight), device=device)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    es_idx = np.asarray(es_idx, dtype=np.int64)
    y_es = arrays.y[es_idx]
    steps_per_epoch = int(np.ceil(len(train_idx) / batch_size))
    sched = _cosine(opt, epochs * steps_per_epoch)
    rng = np.random.RandomState(seed)
    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(train_idx)
        tot, t0 = 0.0, time.time()
        for s in range(steps_per_epoch):
            rows = perm[s * batch_size:(s + 1) * batch_size]
            batch = build_batch(arrays, rows, device)
            y = torch.from_numpy(arrays.y[rows].astype(np.float32)).to(device)
            loss = F.binary_cross_entropy_with_logits(model(batch), y, pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        p = score(model, arrays, es_idx, device=device)
        ap = float(average_precision_score(y_es, p)) if y_es.sum() > 0 else 0.0
        rec = {"epoch": ep + 1, "train_loss": tot / steps_per_epoch, "es_pr_auc": ap,
               "seconds": time.time() - t0}
        history.append(rec)
        print("finetune", json.dumps(rec), flush=True)
        if ap > best:
            best, bad = ap, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


# --------------------------------------------------------------------------- saving

def save_model_and_scores(name, model, ids_valid, scores_valid):
    """Scores via artifacts.save_artifacts, state dict as models/<name>/model.pt."""
    from ecommerce_fraud.artifacts import models_dir, save_artifacts
    out = save_artifacts(name, None, ids_valid, scores_valid, None, None)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, models_dir() / name / "model.pt")
    return out


# --------------------------------------------------------------------------- synthetic smoke

def synthetic_arrays(n=5000, n_num=35, n_cat=30, L=31, pos_rate=0.04, seed=0):
    rng = np.random.RandomState(seed)
    dt = np.cumsum(rng.randint(1, 600, size=n)).astype(np.int64)  # strictly increasing
    cards = rng.randint(3, 40, size=n_cat).astype(np.int32)
    cat = np.stack([rng.randint(0, c, size=n) for c in cards], axis=1).astype(np.int32)
    num = rng.randn(n, n_num).astype(np.float32)
    nan = rng.rand(n, n_num) < 0.1
    num[nan] = 0.0
    logit = 1.5 * num[:, 0] + 0.8 * (cat[:, 0] == 1) + rng.randn(n) * 0.5
    y = (logit > np.quantile(logit, 1 - pos_rate)).astype(np.float32)
    hist = np.full((n, L), -1, dtype=np.int32)
    for i in range(n):
        k = min(i, rng.randint(0, L + 1))
        if k:
            hist[i, L - k:] = np.sort(rng.choice(i, size=k, replace=False))
    return SimpleNamespace(
        ids=np.arange(n, dtype=np.int64) + 1000, dt=dt, y=y, labeled=np.ones(n, bool), num=num,
        num_missing=nan.astype(np.uint8), cat=cat, cat_cardinality=cards, hist=hist,
        num_names=np.array([f"n{i}" for i in range(n_num)]), cat_names=np.array([f"c{i}" for i in range(n_cat)]),
        meta=json.dumps({"synthetic": True}))


def smoke() -> None:
    from ecommerce_fraud.baseline_lgbm import metrics
    device = pick_device()
    a = synthetic_arrays()
    n = len(a.dt)
    print(f"device={device_label(device)} N={n} F={a.num.shape[1]} K={a.cat.shape[1]} L={a.hist.shape[1]} "
          f"pos={a.y.mean():.3f} rows_without_history={(a.hist[:, -1] < 0).sum()}")
    state, plog = pretrain(a, np.arange(n), epochs=1, seed=0, device=device)
    assert np.isfinite(plog[-1]["loss"])
    tr, es, te = np.arange(0, 3000), np.arange(3000, 4000), np.arange(4000, n)
    model, hist = finetune(a, tr, es, epochs=1, seed=0, device=device, init_state=state)
    p = score(model, a, te)
    assert p.shape == (len(te),) and np.isfinite(p).all()
    e = embed(model, a, te[:50])
    assert e.shape == (50, 64)
    print("pretrain loss", [round(r["loss"], 4) for r in plog], "finetune", hist)
    print("metrics", json.dumps(metrics(a.y[te], p)))
    print("SEQ_TRAIN SMOKE OK")


# --------------------------------------------------------------------------- windows

def _test_ids() -> np.ndarray:
    from ecommerce_fraud import data
    return data.time_split(data.load_joined())[2]["TransactionID"].to_numpy(np.int64)


def guard_no_test(a, test_ids, sets: dict) -> None:
    """Raise when any row of any named index set is a test-split TransactionID."""
    for name, idx in sets.items():
        hit = np.isin(a.ids[np.asarray(idx, dtype=np.int64)], test_ids)
        if hit.any():
            raise RuntimeError(f"{name} contains {int(hit.sum())} test-split rows (ADR-002/003 forbid this)")


def select_window(a, window: str, es_days: int, test_ids=None):
    """Returns (fit_end_dt, train_idx, es_idx, score_idx, pretrain_idx) as row indices into the arrays."""
    from ecommerce_fraud import data
    if window == "valid":
        train, valid, test, _ = data.time_split(data.load_joined())
        pos = {int(i): k for k, i in enumerate(a.ids)}
        for nm, part in (("train", train), ("valid", valid)):
            miss = sum(int(i) not in pos for i in part["TransactionID"])
            if miss:
                raise ValueError(f"{miss} {nm} TransactionIDs missing from the arrays")
        tr_rows = np.array([pos[int(i)] for i in train["TransactionID"]], dtype=np.int64)
        score_idx = np.array([pos[int(i)] for i in valid["TransactionID"]], dtype=np.int64)
        fit_end = int(train["TransactionDT"].max())
        es_start = fit_end - es_days * DAY
        train_idx = tr_rows[a.dt[tr_rows] <= es_start]
        es_idx = tr_rows[a.dt[tr_rows] > es_start]
        test_ids = test["TransactionID"].to_numpy(np.int64)
    else:
        m = re.fullmatch(r"fold:(\d+)-(\d+)", window)
        if not m:
            raise ValueError(f"--window must be 'valid' or 'fold:S-E', got {window!r}")
        s_day, e_day = int(m.group(1)), int(m.group(2))
        if s_day - 8 < 1 or e_day < s_day:
            raise ValueError(f"bad fold {window}")
        fit_end = (s_day - 8 + 1) * DAY - 1  # end of day S-8
        es_start = (s_day - 8 - 14 + 1) * DAY - 1  # slice = last 14 train days
        day = a.dt // DAY
        lab = a.labeled
        train_idx = np.where(lab * (a.dt <= es_start))[0]
        es_idx = np.where(lab * (a.dt > es_start) * (a.dt <= fit_end))[0]
        score_idx = np.where(lab * (day >= s_day) * (day <= e_day))[0]
        if test_ids is None:
            test_ids = _test_ids()
    pretrain_idx = np.where(a.dt <= fit_end)[0]
    guard_no_test(a, test_ids, {"scored": score_idx, "fine-tune": train_idx, "early-stopping": es_idx,
                                "pretraining": pretrain_idx})
    if min(len(train_idx), len(es_idx), len(score_idx)) == 0:
        raise ValueError(f"empty set: train={len(train_idx)} es={len(es_idx)} score={len(score_idx)}")
    return fit_end, train_idx, es_idx, score_idx, pretrain_idx


def run(args) -> None:
    from ecommerce_fraud import seq_data
    from ecommerce_fraud.artifacts import WORKTREE as ART_ROOT
    from ecommerce_fraud.baseline_lgbm import metrics
    if args.models_dir:
        os.environ["MODELS_DIR"] = str(Path(args.models_dir).resolve())  # read by artifacts.models_dir()
    from ecommerce_fraud.artifacts import models_dir
    device = pick_device()
    print(f"device={device_label(device)}", flush=True)
    a = seq_data.load_arrays(args.arrays)
    meta = json.loads(a.meta) if isinstance(a.meta, str) else {}
    fit_end, train_idx, es_idx, score_idx, pre_idx = select_window(a, args.window, args.es_days)
    if meta.get("fit_end_dt") is None or int(meta["fit_end_dt"]) != fit_end:
        raise ValueError(f"window {args.window} needs fit_end_dt={fit_end} but arrays meta has "
                         f"{meta.get('fit_end_dt')}; normalization must be fitted on the same fit window. "
                         f"Rebuild with seq_data --full using fit_end_dt={fit_end} "
                         f"(--fit-end-day gives (day+1)*86400-1; day {fit_end // DAY} gives "
                         f"{(fit_end // DAY + 1) * DAY - 1})")
    print(f"window={args.window} fit_end_dt={fit_end} pretrain={len(pre_idx)} finetune={len(train_idx)} "
          f"es={len(es_idx)} score={len(score_idx)} test_rows_in_any_set=0 device={device}")
    out_dir = models_dir() / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir = (Path(args.results_dir).resolve() if args.results_dir else ART_ROOT / "results") / args.name
    res_dir.mkdir(parents=True, exist_ok=True)
    info = {"name": args.name, "mode": args.mode, "seed": args.seed, "window": args.window,
            "fit_end_dt": fit_end, "es_days": args.es_days, "arrays_meta": meta}
    counts = {"n_pretrain": len(pre_idx), "n_finetune": len(train_idx), "n_es": len(es_idx),
              "n_score": len(score_idx)}
    info.update(counts)
    timing = {"device": device_label(device)}
    state = None
    if args.mode in ("pretrain", "ssl_finetune"):
        t0 = time.time()
        state, plog = pretrain(a, pre_idx, args.pretrain_epochs, args.seed, device)
        timing["seconds_pretrain"] = time.time() - t0
        torch.save(state, out_dir / "pretrain.pt")
        info["pretrain_log"] = plog
    result = {}
    if args.mode in ("finetune", "ssl_finetune"):
        t0 = time.time()
        model, hist = finetune(a, train_idx, es_idx, args.finetune_epochs, args.seed, device, init_state=state)
        timing["seconds_finetune"] = time.time() - t0
        p = score(model, a, score_idx)
        save_model_and_scores(args.name, model, a.ids[score_idx], p)
        info["finetune_history"] = hist
        timing["best_epoch"] = 1 + int(np.argmax([h["es_pr_auc"] for h in hist]))  # first max = the kept state
        result = metrics(a.y[score_idx], p)
        print(json.dumps(result))
        if args.save_embed:
            emb_idx = np.concatenate([train_idx, es_idx, score_idx])
            test_ids = _test_ids()
            guard_no_test(a, test_ids, {"embedding": emb_idx})
            ids = a.ids[emb_idx].astype(np.int64)
            assert len(np.unique(ids)) == len(ids)
            emb = embed(model, a, emb_idx, device=device)
            assert emb.shape == (len(ids), 64) and emb.dtype == np.float32 and np.isfinite(emb).all()
            assert not np.isin(ids, test_ids).any()
            np.savez(out_dir / "embed.npz", ids=ids, emb=emb)
            print(f"embed.npz ids={ids.shape} emb={emb.shape}")
    info.update(timing)
    (res_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    (res_dir / "metrics.json").write_text(json.dumps({**result, **counts, **timing}, indent=2))


# --------------------------------------------------------------------------- real-data smoke

def smoke_real() -> None:
    from ecommerce_fraud import seq_data
    from ecommerce_fraud.baseline_lgbm import metrics
    path = seq_data.data_mod.WORKTREE / "data" / "processed" / "seq" / "smoke.npz"
    if not path.exists():
        seq_data._smoke()
    device = pick_device()
    a = seq_data.load_arrays(path)
    fit_end = int(json.loads(a.meta)["fit_end_dt"])
    fit = a.dt <= fit_end
    es_start = fit_end - 2 * DAY
    pre_idx = np.where(fit)[0]
    train_idx = np.where(fit * a.labeled * (a.dt <= es_start))[0]
    es_idx = np.where(fit * a.labeled * (a.dt > es_start))[0]
    score_idx = np.where(a.labeled * (a.dt > fit_end))[0]
    guard_no_test(a, _test_ids(), {"scored": score_idx, "fine-tune": train_idx, "early-stopping": es_idx,
                                   "pretraining": pre_idx})
    print(f"device={device} arrays={len(a.dt)} fit_end_dt={fit_end} pretrain={len(pre_idx)} "
          f"finetune={len(train_idx)} es={len(es_idx)} score={len(score_idx)} "
          f"es_pos={int(a.y[es_idx].sum())} score_pos={int(a.y[score_idx].sum())}")
    state, plog = pretrain(a, pre_idx, epochs=1, seed=0, device=device)
    assert np.isfinite(plog[-1]["loss"])
    model, hist = finetune(a, train_idx, es_idx, epochs=1, seed=0, device=device, init_state=state)
    p = score(model, a, score_idx)
    assert p.shape == (len(score_idx),) and np.isfinite(p).all()
    m = metrics(a.y[score_idx], p)
    e = embed(model, a, score_idx[:10])
    print(f"pretrain loss={plog[-1]['loss']:.4f} (cat {plog[-1]['cat_loss']:.4f}, num {plog[-1]['num_loss']:.4f}) "
          f"seconds/epoch={plog[-1]['seconds']:.1f} rows={len(pre_idx)}")
    print(f"finetune train_loss={hist[-1]['train_loss']:.4f} es_pr_auc={hist[-1]['es_pr_auc']:.4f} "
          f"seconds/epoch={hist[-1]['seconds']:.1f} rows={len(train_idx)}")
    print("metrics", json.dumps(m))
    print(f"embed shape={tuple(e.shape)} (no 16-dim projection exists; using encoder d_model={model.encoder.d_model})")
    print("SEQ_TRAIN REAL SMOKE OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-synthetic", action="store_true")
    ap.add_argument("--smoke-real", action="store_true")
    ap.add_argument("--arrays")
    ap.add_argument("--mode", choices=["pretrain", "finetune", "ssl_finetune"])
    ap.add_argument("--name")
    ap.add_argument("--window", help="valid | fold:S-E (required for full runs)")
    ap.add_argument("--es-days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrain-epochs", type=int, default=5)
    ap.add_argument("--finetune-epochs", type=int, default=15)
    ap.add_argument("--save-embed", action="store_true", help="ssl_finetune only: write models/<name>/embed.npz")
    ap.add_argument("--models-dir", help="override the models/ root (default: artifacts.models_dir())")
    ap.add_argument("--results-dir", help="override the results/ root (default: <repo>/results)")
    args = ap.parse_args()
    if args.smoke_synthetic:
        return smoke()
    if args.smoke_real:
        return smoke_real()
    if not (args.arrays and args.mode and args.name and args.window):
        ap.error("--arrays, --mode, --name and --window are required")
    if args.save_embed and args.mode != "ssl_finetune":
        ap.error("--save-embed requires --mode ssl_finetune")
    run(args)


if __name__ == "__main__":
    main()
