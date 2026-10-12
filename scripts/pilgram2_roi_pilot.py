"""FF++ c0 + five fixed pilgram2 filters: prepare, train and paired evaluation."""
import argparse, copy, hashlib, importlib.metadata, json, math, random, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import pilgram2
import timm
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from sklearn.metrics import roc_auc_score
from train_baseline import LABEL_MAP, classification_metrics, sha256_file, write_json
from xception_preprocessing import crop_face_roi, evaluation_transform_from_checkpoint, interpolation_mode

METHODS = ("original", "Deepfakes", "Face2Face")
STRATEGIES = ("source_ft", "augmentation", "dann")
def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_json(tmp, value)
    tmp.replace(path)
def atomic_checkpoint(path, value):
    tmp = path.with_suffix(".tmp.pt")
    torch.save(value, tmp)
    tmp.replace(path)
def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
def cache_path(root, condition, row):
    return root / "images" / condition / row["method"] / (Path(row["source_path"]).stem + ".png")
def read_base(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["config"]["preprocessing_name"] == "face_roi_mtcnn_1p5_299"
    assert checkpoint["config"]["manifest_sha256"] == sha256_file(args.manifest)
    assert checkpoint["num_classes"] == 2
    return checkpoint
def prepare(args, protocol):
    base = read_base(args)
    frame = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    assert frame.source_path.is_unique
    assert frame.groupby("group_id")["split"].nunique().max() == 1
    pieces = []
    for split, limit in (("train", args.train_per_method), ("val", args.val_per_method), ("test", args.test_per_method)):
        for method in METHODS:
            part = frame[(frame.split == split) & (frame.method == method)]
            assert len(part) > 0
            if limit:
                assert len(part) >= limit, (split, method, limit)
                part = part.sample(n=limit, random_state=args.seed)
            pieces.append(part.sort_values("source_path"))
    subset = pd.concat(pieces, ignore_index=True)
    assert all(LABEL_MAP[r.label] == (0 if r.method == "original" else 1) for r in subset.itertuples())
    manifest_text = subset.to_csv(index=False)
    config = {
        "source_manifest_sha256": sha256_file(args.manifest),
        "initial_checkpoint_sha256": sha256_file(args.checkpoint),
        "pilot_manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
        "protocol": protocol, "seed": args.seed,
        "pilgram2_version": importlib.metadata.version("pilgram2"),
        "counts": subset.groupby(["split", "method"]).size().to_dict(),
        "data_config": base["config"]["data_config"],
    }
    config["counts"] = {f"{s}/{m}": int(n) for (s, m), n in config["counts"].items()}
    config["identity"] = key(config)
    cfg_path = args.cache_root / "config.json"
    if cfg_path.exists():
        assert json.loads(cfg_path.read_text())["identity"] == config["identity"], "Cache protocol differs: choose a new cache root"
    args.cache_root.mkdir(parents=True, exist_ok=True)
    atomic_json(cfg_path, config)
    manifest_path = args.cache_root / "pilot_manifest.csv"
    if manifest_path.exists():
        assert sha256_file(manifest_path) == config["pilot_manifest_sha256"]
    else:
        manifest_path.write_text(manifest_text, encoding="utf-8")
    rows = subset.to_dict("records")
    missing = [r["source_path"] for r in rows if not (args.data_root / r["source_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing source images: {missing[:3]}")
    resize = transforms.Resize([299, 299], interpolation=interpolation_mode(config["data_config"]["interpolation"]), antialias=True)
    def one(row):
        conditions = ["normal"] + (protocol["filters"] if row["split"] != "val" else [])
        pending = [c for c in conditions if not cache_path(args.cache_root, c, row).is_file()]
        if not pending:
            return
        with Image.open(args.data_root / row["source_path"]) as opened:
            original = opened.convert("RGB")
            for condition in pending:
                image = original if condition == "normal" else getattr(pilgram2, condition)(original.copy()).convert("RGB")
                image = resize(crop_face_roi(image, row, fallback_size=299))
                dest = cache_path(args.cache_root, condition, row)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".tmp.png")
                image.save(tmp, format="PNG")
                tmp.replace(dest)
    print(f"Preparing {len(rows)} source images; cached filtered ROI PNGs, no JPEG", flush=True)
    last = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers or 1) as pool:
        for done, _ in enumerate(pool.map(one, rows), 1):
            if done == len(rows) or time.monotonic() - last >= 30:
                print(f"Cache: {done}/{len(rows)} ({done / len(rows):.0%})", flush=True)
                last = time.monotonic()
    atomic_json(args.cache_root / "ready.json", {"identity": config["identity"], "images": len(rows)})
    print("Cache ready:", config["counts"], flush=True)

class Views(Dataset):
    def __init__(self, frame, root, config, filters, seed, epoch=0, strategy=None, condition="normal"):
        self.rows = frame.to_dict("records")
        self.root, self.filters = root, filters
        self.seed, self.epoch, self.strategy, self.condition = seed, epoch, strategy, condition
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(config["mean"], config["std"])])
    def __len__(self):
        return len(self.rows)
    def image(self, row, condition, flip=False):
        with Image.open(cache_path(self.root, condition, row)) as opened:
            image = opened.convert("RGB")
            if flip:
                image = TF.hflip(image)
            return self.transform(image)
    def __getitem__(self, index):
        row = self.rows[index]
        y = LABEL_MAP[row["label"]]
        if self.strategy is None:
            return self.image(row, self.condition), y
        rng = random.Random(self.seed + self.epoch * 1000003 + index)
        condition, flip = rng.choice(self.filters), rng.random() < 0.5
        xs = self.image(row, "normal", flip)
        xt = xs.clone() if self.strategy == "source_ft" else self.image(row, condition, flip)
        return xs, xt, y

class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None

def model_from(checkpoint):
    model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model
def load_cache(args, protocol):
    cfg = json.loads((args.cache_root / "config.json").read_text())
    assert cfg["protocol"] == protocol and cfg["seed"] == args.seed
    assert cfg["source_manifest_sha256"] == sha256_file(args.manifest)
    assert cfg["initial_checkpoint_sha256"] == sha256_file(args.checkpoint)
    assert json.loads((args.cache_root / "ready.json").read_text())["identity"] == cfg["identity"]
    assert sha256_file(args.cache_root / "pilot_manifest.csv") == cfg["pilot_manifest_sha256"]
    frame = pd.read_csv(args.cache_root / "pilot_manifest.csv", dtype=str, keep_default_na=False)
    return cfg, frame
def loader(dataset, args, shuffle=False, epoch=0):
    generator = torch.Generator().manual_seed(args.seed + epoch)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, persistent_workers=False,
                      pin_memory=args.device == "cuda", generator=generator)
@torch.no_grad()
def scores(model, data_loader, args, description):
    model.eval()
    result, labels = [], []
    last = time.monotonic()
    for x, y in data_loader:
        with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=args.device == "cuda"):
            logits = model(x.to(args.device))
        result.extend(logits.softmax(1)[:, 1].float().cpu().tolist())
        labels.extend(y.tolist())
        if time.monotonic() - last >= 30:
            print(f"{description}: {len(result)}/{len(data_loader.dataset)}", flush=True)
            last = time.monotonic()
    return np.asarray(labels), np.asarray(result)
def metrics(frame, y, p):
    result = classification_metrics(y, (p >= .5).astype(int), p)
    result["real_fpr"] = float((p[y == 0] >= .5).mean())
    result["fake_fnr"] = float((p[y == 1] < .5).mean())
    for method in METHODS[1:]:
        use = (frame.method.to_numpy() == method) | (y == 0)
        result[method + "_auc"] = float(roc_auc_score(y[use], p[use]))
    result["method_macro_auc"] = (result["Deepfakes_auc"] + result["Face2Face_auc"]) / 2
    return result

def train(args, protocol):
    cache, frame = load_cache(args, protocol)
    initial = read_base(args)
    train_frame, val = frame[frame.split == "train"].copy(), frame[frame.split == "val"].copy()
    cfg = cache["data_config"]
    val_loader = loader(Views(val, args.cache_root, cfg, protocol["filters"], args.seed), args)
    weights = initial["config"]["class_weights"]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([weights["real"], weights["fake"]], device=args.device))
    for strategy in args.strategies:
        if strategy == "baseline":
            continue
        assert strategy in STRATEGIES
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        identity = key({"cache": cache["identity"], "strategy": strategy, "epochs": args.epochs,
                        "batch_size": args.batch_size, "lr": args.learning_rate,
                        "weight_decay": args.weight_decay, "domain_weight": args.domain_weight})
        folder = args.runs_root / strategy
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "completed.json").exists():
            assert json.loads((folder / "completed.json").read_text())["identity"] == identity
            assert (folder / "best.pt").is_file()
            print("Skipping completed:", strategy, flush=True)
            continue
        model = model_from(initial).to(args.device)
        discriminator = nn.Sequential(nn.Linear(model.num_features, 128), nn.ReLU(), nn.Linear(128, 1)).to(args.device)
        parameters = list(model.parameters()) + (list(discriminator.parameters()) if strategy == "dann" else [])
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=args.device == "cuda")
        start, best, history = 0, -1.0, []
        if (folder / "last.pt").exists():
            saved = torch.load(folder / "last.pt", map_location=args.device, weights_only=False)
            assert saved["pilot_identity"] == identity
            model.load_state_dict(saved["model_state_dict"])
            discriminator.load_state_dict(saved["discriminator_state_dict"])
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            scaler.load_state_dict(saved["scaler_state_dict"])
            start, best, history = saved["epoch"], saved["best_val_auc"], saved["history"]
            print(f"Resuming {strategy} after epoch {start}", flush=True)
        run_config = copy.deepcopy(initial["config"])
        run_config.update(experiment_family=protocol["name"], condition_name=strategy,
                          manifest_sha256=cache["source_manifest_sha256"], pilot_manifest_sha256=cache["pilot_manifest_sha256"],
                          train_images=len(train_frame), val_images=len(val), seed=args.seed,
                          learning_rate=args.learning_rate, epochs=args.epochs, domain_weight=args.domain_weight,
                          initial_checkpoint_sha256=cache["initial_checkpoint_sha256"], pilot_identity=identity)
        run_config["class_counts"] = {"real": int((train_frame.label == "real").sum()), "fake": int((train_frame.label == "fake").sum())}
        atomic_json(folder / "config.json", run_config)
        for epoch in range(start, args.epochs):
            ds = Views(train_frame, args.cache_root, cfg, protocol["filters"], args.seed, epoch, strategy)
            batches = loader(ds, args, True, epoch)
            model.train(); discriminator.train()
            sums, done, last = np.zeros(3), 0, time.monotonic()
            print(f"{strategy}: epoch {epoch+1}/{args.epochs}", flush=True)
            for step, (xs, xt, y) in enumerate(batches):
                xs, xt, y = xs.to(args.device), xt.to(args.device), y.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                progress = (epoch * len(batches) + step) / max(1, args.epochs * len(batches) - 1)
                alpha = 2 / (1 + math.exp(-10 * progress)) - 1
                with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=args.device == "cuda"):
                    features = model.forward_features(torch.cat([xs, xt]))
                    logits = model.forward_head(features)
                    n = len(y)
                    cls = criterion(logits[:n], y)
                    if strategy == "augmentation":
                        cls = .5 * (cls + criterion(logits[n:], y))
                    domain = cls.new_zeros(())
                    if strategy == "dann":
                        pooled = features.mean(dim=(2, 3))
                        domain_logits = discriminator(Reverse.apply(pooled, alpha)).flatten()
                        domain_labels = torch.cat([torch.zeros(n, device=args.device), torch.ones(n, device=args.device)])
                        domain = nn.functional.binary_cross_entropy_with_logits(domain_logits, domain_labels)
                    loss = cls + args.domain_weight * domain
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                sums += np.asarray([loss.item(), cls.item(), domain.item()]) * n
                done += n
                if time.monotonic() - last >= 30:
                    print(f"  train {done}/{len(ds)} | loss={sums[0]/done:.4f}", flush=True)
                    last = time.monotonic()
            yv, pv = scores(model, val_loader, args, "Validation normal")
            auc = float(roc_auc_score(yv, pv))
            history.append({"epoch": epoch+1, "loss": sums[0]/done, "classification_loss": sums[1]/done,
                            "domain_loss": sums[2]/done, "validation_original_auc": auc})
            improved = auc > best
            best = max(best, auc)
            state = {"model_name": initial["model_name"], "num_classes": 2, "model_state_dict": model.state_dict(),
                     "config": run_config, "epoch": epoch+1, "best_val_auc": best, "pilot_identity": identity}
            if improved:
                atomic_checkpoint(folder / "best.pt", state)
            atomic_checkpoint(folder / "last.pt", dict(state, optimizer_state_dict=optimizer.state_dict(),
                              discriminator_state_dict=discriminator.state_dict(), scaler_state_dict=scaler.state_dict(), history=history))
            atomic_json(folder / "history.json", history)
            print(f"{strategy}: val original AUC={auc:.4f}; best={best:.4f}", flush=True)
        atomic_json(folder / "completed.json", {"identity": identity, "epochs": args.epochs, "best_val_auc": best})
        del model, discriminator, optimizer
        if args.device == "cuda":
            torch.cuda.empty_cache()

def evaluate(args, protocol):
    cache, frame = load_cache(args, protocol)
    test = frame[frame.split == "test"].reset_index(drop=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for strategy in args.strategies:
        checkpoint_path = args.checkpoint if strategy == "baseline" else args.runs_root / strategy / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if strategy != "baseline":
            assert checkpoint["config"]["pilot_manifest_sha256"] == cache["pilot_manifest_sha256"]
            assert checkpoint["config"]["initial_checkpoint_sha256"] == cache["initial_checkpoint_sha256"]
            assert checkpoint["config"]["condition_name"] == strategy
        model = model_from(checkpoint).to(args.device)
        checkpoint_sha = sha256_file(checkpoint_path)
        for condition in ["normal"] + protocol["filters"]:
            identity = key({"cache": cache["identity"], "checkpoint": checkpoint_sha, "condition": condition})
            out = args.output_root / strategy / condition
            out.mkdir(parents=True, exist_ok=True)
            saved_path = out / "metrics.json"
            if saved_path.exists():
                saved = json.loads(saved_path.read_text())
                assert saved["identity"] == identity, "Evaluation identity differs: choose a new output root"
                assert (out / "predictions.csv").is_file()
                result = saved["metrics"]
                print("Skipping completed evaluation:", strategy, condition, flush=True)
            else:
                print(f"Evaluating {strategy} / {condition} / {len(test)} images", flush=True)
                dl = loader(Views(test, args.cache_root, cache["data_config"], protocol["filters"], args.seed, condition=condition), args)
                y, p = scores(model, dl, args, f"{strategy}/{condition}")
                result = metrics(test, y, p)
                predictions = test[["source_path", "split", "label", "method", "group_id", "video_id"]].copy()
                predictions["condition"] = condition
                predictions["fake_probability"] = p
                predictions["predicted_class"] = (p >= .5).astype(int)
                predictions.to_csv(out / "predictions.csv", index=False)
                atomic_json(saved_path, {"identity": identity, "checkpoint_sha256": checkpoint_sha,
                                         "images": len(test), "threshold": .5, "metrics": result})
            summary.append(dict(strategy=strategy, condition=condition, images=len(test),
                                **{k:v for k,v in result.items() if k != "confusion_matrix"}))
            print(f"  AUC={result['roc_auc']:.4f}, Real FPR={result['real_fpr']:.2%}, Fake FNR={result['fake_fnr']:.2%}", flush=True)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    table = pd.DataFrame(summary)
    table.to_csv(args.output_root / "condition_summary.csv", index=False)
    aggregate = []
    for strategy, part in table.groupby("strategy", sort=False):
        clean, filtered = part[part.condition == "normal"].iloc[0], part[part.condition != "normal"]
        aggregate.append({"strategy": strategy, "original_auc": clean.roc_auc, "original_real_fpr": clean.real_fpr,
                          "filtered_mean_auc": filtered.roc_auc.mean(), "filtered_worst_auc": filtered.roc_auc.min(),
                          "filtered_mean_real_fpr": filtered.real_fpr.mean(), "filtered_mean_fake_fnr": filtered.fake_fnr.mean()})
    pd.DataFrame(aggregate).to_csv(args.output_root / "model_summary.csv", index=False)
    if "baseline" in set(table.strategy):
        ref = table[table.strategy == "baseline"][["condition", "roc_auc", "real_fpr", "fake_fnr"]]
        delta = table.merge(ref, on="condition", suffixes=("", "_baseline"))
        for name in ("roc_auc", "real_fpr", "fake_fnr"):
            delta[name + "_delta"] = delta[name] - delta[name + "_baseline"]
        delta.to_csv(args.output_root / "baseline_deltas.csv", index=False)
    print(pd.DataFrame(aggregate).to_string(index=False), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "train", "evaluate"])
    for name in ("data-root", "manifest", "checkpoint", "cache-root"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1]/"configs/pilgram2_roi_pilot_v1.json")
    p.add_argument("--runs-root", type=Path, default=Path("outputs/pilgram2_roi_pilot_v1/runs"))
    p.add_argument("--output-root", type=Path, default=Path("outputs/pilgram2_roi_pilot_v1/evaluation"))
    p.add_argument("--strategies", nargs="+", choices=["baseline", *STRATEGIES], default=["baseline", *STRATEGIES])
    p.add_argument("--train-per-method", type=int, default=1000)
    p.add_argument("--val-per-method", type=int, default=300)
    p.add_argument("--test-per-method", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--domain-weight", type=float, default=.1)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    for n in ("train_per_method", "val_per_method", "test_per_method", "workers"):
        assert getattr(args,n) >= 0, n
    assert args.epochs > 0 and args.batch_size > 0 and args.learning_rate > 0
    protocol = json.loads(args.protocol.read_text())
    assert all(callable(getattr(pilgram2, name, None)) for name in protocol["filters"])
    if args.action == "prepare": prepare(args, protocol)
    elif args.action == "train": train(args, protocol)
    else: evaluate(args, protocol)
if __name__ == "__main__":
    main()
