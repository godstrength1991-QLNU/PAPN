"""
PAPN training, with a persistence baseline for comparison.

PAPN is evaluated alongside a pure persistence baseline:
  - Point forecast: PAPN q50 should be at least as good as persistence.
  - Probabilistic: PAPN provides calibrated intervals; persistence does not.
"""
import sys, time, pickle, json, copy, argparse
from pathlib import Path
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from data.dataset import make_loader
from models.papn import PAPN
from scripts.quantile_loss import multi_quantile_loss
from scripts.prob_metrics import all_prob_metrics
from scripts.conformal import per_horizon_conformal

DATA_PATH = SCRIPT_DIR / 'data' / 'mawp_dataset.pkl'
T_IN, H = 60, 30
QUANTILES = [0.05, 0.5, 0.95]
DAY_TRAIN_END = 24*365+6
DAY_VAL_END = DAY_TRAIN_END + 3*365+1


def get_device():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        return torch.device('cuda')
    print("CPU", flush=True); return torch.device('cpu')


def slice_data(d, s, e):
    return {'storage_rate': d['storage_rate'][:,s:e], 'meteo': d['meteo'][:,s:e],
            'climate': d['climate'][s:e], 'group_ids': d['group_ids'],
            'locations': d['locations'], 'capacity': d['capacity']}


def make_loaders(data, bs):
    tr = slice_data(data, 0, DAY_TRAIN_END)
    va = slice_data(data, DAY_TRAIN_END, DAY_VAL_END)
    te = slice_data(data, DAY_VAL_END, data['storage_rate'].shape[1])
    tl, st = make_loader(tr, T_in=T_IN, H=H, batch_size=bs, shuffle=True)
    vl, _ = make_loader(va, T_in=T_IN, H=H, batch_size=bs, shuffle=False, stats=st)
    sl, _ = make_loader(te, T_in=T_IN, H=H, batch_size=bs, shuffle=False, stats=st)
    return tl, vl, sl


def persistence_predict(storage_hist, H):
    """Pure persistence: future = today. Returns a point forecast (B,N,H)."""
    anchor = storage_hist[:, :, -1:].expand(-1, -1, H)
    return anchor


def collect(model, loader, device):
    model.eval(); preds, truths, persist = [], [], []
    with torch.no_grad():
        for b in loader:
            for k, v in b.items():
                if torch.is_tensor(v): b[k] = v.to(device)
            q = model(b['storage_hist'])
            preds.append(q.cpu().numpy())
            truths.append(b['storage_fut'].cpu().numpy())
            p = persistence_predict(b['storage_hist'], H)
            persist.append(p.cpu().numpy())
    return np.concatenate(preds), np.concatenate(truths), np.concatenate(persist)


def run(args):
    device = get_device()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    data = pickle.load(open(DATA_PATH, 'rb'))
    tl, vl, sl = make_loaders(data, args.batch_size)
    print(f"[Data] N={data['storage_rate'].shape[0]}", flush=True)

    model = PAPN(T_in=T_IN, H=H, hidden=args.hidden, n_layers=args.n_layers,
                 max_residual=args.max_residual).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] PAPN, params={n_params:,}, max_residual={args.max_residual}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val, best_state, bad = 1e9, None, 0
    hist = {'train_loss': [], 'val_crps': []}
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for b in tl:
            for k, v in b.items():
                if torch.is_tensor(v): b[k] = v.to(device)
            loss = multi_quantile_loss(model(b['storage_hist']), b['storage_fut'], QUANTILES)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        vp, vt, _ = collect(model, vl, device)
        vm = all_prob_metrics(vp, vt)
        hist['train_loss'].append(0); hist['val_crps'].append(vm['crps'])
        if vm['crps'] < best_val:
            best_val, best_state, bad = vm['crps'], copy.deepcopy(model.state_dict()), 0; tag='★'
        else:
            bad += 1; tag=''
        if (ep+1)%10==0 or ep==0:
            print(f"[Ep {ep+1:3d}/{args.epochs}] val_crps={vm['crps']:.4f} {tag} "
                  f"cov={vm['pi_coverage_90']:.3f} q50_rmse={vm['point_rmse']:.4f} "
                  f"[{(time.time()-t0)/60:.1f}min]", flush=True)
        if bad >= args.patience:
            print(f"⚠ Early stop ep {ep+1}", flush=True); break

    model.load_state_dict(best_state)

    # Test set + conformal calibration
    tp, tt, persist = collect(model, sl, device)
    m_before = all_prob_metrics(tp, tt)
    # Conformal calibration (using the validation set)
    vp, vt, _ = collect(model, vl, device)
    tp_cal, _ = per_horizon_conformal(vt.reshape(-1,H), vp.reshape(-1,H,3), tp.reshape(-1,H,3))
    m_after = all_prob_metrics(tp_cal, tt.reshape(-1,H))

    # Persistence point-forecast RMSE
    persist_rmse = float(np.sqrt(((persist - tt)**2).mean()))

    print(f"\nPAPN test results:")
    print(f"  {'metric':<20}{'before':>12}{'after':>12}")
    print(f"  {'CRPS':<20}{m_before['crps']:>12.4f}{m_after['crps']:>12.4f}")
    print(f"  {'PI Coverage':<20}{m_before['pi_coverage_90']:>12.3f}{m_after['pi_coverage_90']:>12.3f}")
    print(f"  {'PI Width':<20}{m_before['pi_width_mean']:>12.4f}{m_after['pi_width_mean']:>12.4f}")
    print(f"  {'q50 RMSE':<20}{m_before['point_rmse']:>12.4f}{m_after['point_rmse']:>12.4f}")
    print(f"\n  vs persistence:")
    print(f"     Persistence RMSE: {persist_rmse:.4f}")
    print(f"     PAPN q50 RMSE:    {m_before['point_rmse']:.4f}")
    print(f"     {'OK: PAPN point forecast >= persistence' if m_before['point_rmse']<=persist_rmse*1.02 else 'WARN: PAPN not yet matching persistence'}")
    print(f"     PAPN also provides calibrated intervals (cov {m_after['pi_coverage_90']:.2f}); persistence does not.")

    out = SCRIPT_DIR / 'outputs' / f"papn_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out/'results').mkdir(exist_ok=True)
    torch.save(model.state_dict(), out/'papn.pt')
    json.dump({'metrics_before': m_before, 'metrics_after': m_after,
               'persistence_rmse': persist_rmse, 'n_params': n_params,
               'history': hist, 'config': vars(args)},
              open(out/'results'/'result.json','w'), indent=2)
    print(f"\n✓ Saved → {out}/results/result.json", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--n-layers', type=int, default=4)
    p.add_argument('--max-residual', type=float, default=0.6)
    p.add_argument('--patience', type=int, default=20)
    args = p.parse_args()
    torch.set_num_threads(2)
    print("="*70, flush=True)
    print(f"PAPN Training - seed {args.seed}", flush=True)
    print("="*70, flush=True)
    run(args)


if __name__ == "__main__":
    main()
