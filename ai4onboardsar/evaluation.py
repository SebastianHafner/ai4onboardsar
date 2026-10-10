"""Streaming inference over whole scenes, metrics, and the non-learned threshold baseline."""
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from .models import CausalTCN


@torch.no_grad()
def stream_predict(model, x: np.ndarray, delay: int, chunk: int = 256) -> np.ndarray:
    """Run a causal model over all rows of x (R, C, W) and return water probability per row,
    aligned to labels: prob[r] = sigmoid(output at step r + delay). Last `delay` rows are NaN."""
    model.eval()
    R = x.shape[0]
    out = np.full((R, x.shape[2]), np.nan, np.float32)
    xt = torch.from_numpy(x)
    if not isinstance(model, CausalTCN):  # recurrent: carry the state across chunks
        state = None
        for s in range(0, R, chunk):
            logits, state = model(xt[None, s:s + chunk], state)
            out[s:s + logits.shape[1]] = torch.sigmoid(logits[0]).numpy()
    else:  # stateless causal conv: process with overlapping history
        ctx = model.receptive_field
        for s in range(0, R, chunk):
            a = max(0, s - ctx)
            logits, _ = model(xt[None, a:s + chunk])
            out[s:s + chunk] = torch.sigmoid(logits[0, s - a:]).numpy()
    aligned = np.full_like(out, np.nan)
    aligned[:R - delay] = out[delay:]
    return aligned


def metrics(prob: np.ndarray, label: np.ndarray, thr: float = 0.5) -> dict:
    m = np.isfinite(prob) & np.isfinite(label)
    p, y = prob[m] >= thr, label[m] >= 0.5
    tp, fp, fn = np.sum(p & y), np.sum(p & ~y), np.sum(~p & y)
    tn = np.sum(~p & ~y)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"iou": float(tp / max(tp + fp + fn, 1)), "f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
            "precision": float(prec), "recall": float(rec), "accuracy": float((tp + tn) / max(m.sum(), 1)),
            "water_frac_label": float(y.mean()), "water_frac_pred": float(p.mean()), "n": int(m.sum())}


def threshold_baseline_score(x: np.ndarray, delay: int) -> np.ndarray:
    """Darkness score from the normalized VV channel, averaged over the 2*delay+1 rows centered on
    each row (i.e. causal with latency `delay`). Higher = more likely water. Aligned to labels."""
    vv = x[:, 0]
    return -uniform_filter1d(vv, size=2 * delay + 1, axis=0, mode="nearest")


def fit_threshold(score: np.ndarray, label: np.ndarray) -> float:
    m = np.isfinite(score) & np.isfinite(label)
    s, y = score[m], label[m] >= 0.5
    best, best_t = -1, 0.0
    for t in np.quantile(s, np.linspace(0.5, 0.999, 200)):
        p = s >= t
        tp, fp, fn = np.sum(p & y), np.sum(p & ~y), np.sum(~p & y)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best:
            best, best_t = f1, float(t)
    return best_t
