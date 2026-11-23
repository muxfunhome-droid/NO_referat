import argparse
import json
import os
import pickle
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset_ns_128"
CHECKPOINT_PATH = BASE_DIR / "FNO.pkl"
WIDTH, MODES1, MODES2, LAYERS = 32, 16, 16, 4
PROJ = 128

EPOCHS = 150
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
PAIRS_PER_EPOCH = 1024
VAL_PAIRS_PER_EPOCH = 512
ROLLOUT_CHECK_EVERY = 10
ROLLOUT_VAL_SAMPLES = 1

MODEL_CONFIG = {
    "width": WIDTH,
    "modes1": MODES1,
    "modes2": MODES2,
    "layers": LAYERS,
    "projection": PROJ,
    "input_channels": 1,
    "output_channels": 1,
}


def init_fno(key):
    ks = jax.random.split(key, 3 + 5 * LAYERS)

    p = {
        "P": jax.random.normal(ks[0], (1, WIDTH)) / np.sqrt(1),
        "Pb": jnp.zeros(WIDTH),
        "Q1": jax.random.normal(ks[1], (WIDTH, PROJ)) / np.sqrt(WIDTH),
        "Q1b": jnp.zeros(PROJ),
        "Q2": 1e-3 * jax.random.normal(ks[2], (PROJ, 1)) / np.sqrt(PROJ),
        "Q2b": jnp.zeros(1),
        "L": [],
    }

    sc = 1.0 / (WIDTH * WIDTH)

    for l in range(LAYERS):
        k = 3 + 5 * l
        p["L"].append({
            "R1r": sc * jax.random.uniform(
                ks[k], (MODES1, MODES2, WIDTH, WIDTH), minval=-1.0, maxval=1.0
            ),
            "R1i": sc * jax.random.uniform(
                ks[k + 1], (MODES1, MODES2, WIDTH, WIDTH), minval=-1.0, maxval=1.0
            ),
            "R2r": sc * jax.random.uniform(
                ks[k + 2], (MODES1, MODES2, WIDTH, WIDTH), minval=-1.0, maxval=1.0
            ),
            "R2i": sc * jax.random.uniform(
                ks[k + 3], (MODES1, MODES2, WIDTH, WIDTH), minval=-1.0, maxval=1.0
            ),
            "W": jax.random.normal(ks[k + 4], (WIDTH, WIDTH)) / np.sqrt(WIDTH),
            "b": jnp.zeros(WIDTH),
        })

    return p


def spectral_conv2d(lp, v):
    n1, n2 = v.shape[1], v.shape[2]

    vh = jnp.fft.rfft2(v, axes=(1, 2))
    out = jnp.zeros_like(vh)

    R1 = lp["R1r"] + 1j * lp["R1i"]
    R2 = lp["R2r"] + 1j * lp["R2i"]

    out1 = jnp.einsum(
        "bxyc,xyco->bxyo",
        vh[:, :MODES1, :MODES2, :],
        R1,
    )

    out2 = jnp.einsum(
        "bxyc,xyco->bxyo",
        vh[:, -MODES1:, :MODES2, :],
        R2,
    )

    out = out.at[:, :MODES1, :MODES2, :].set(out1)
    out = out.at[:, -MODES1:, :MODES2, :].set(out2)

    return jnp.fft.irfft2(out, s=(n1, n2), axes=(1, 2))


def fno(p, a):
    v = a[..., None] @ p["P"] + p["Pb"]

    for l, lp in enumerate(p["L"]):
        v = spectral_conv2d(lp, v) + v @ lp["W"] + lp["b"]

        if l < LAYERS - 1:
            v = jax.nn.gelu(v)

    h = jax.nn.gelu(v @ p["Q1"] + p["Q1b"])
    delta = (h @ p["Q2"] + p["Q2b"])[..., 0]
    return a + delta


def rel_l2(pred, y):
    num = jnp.linalg.norm(pred - y, axis=(1, 2))
    den = jnp.linalg.norm(y, axis=(1, 2)) + 1e-8
    return num / den


@jax.jit
def eval_step(params, x, y):
    return rel_l2(fno(params, x), y)


def load_dataset(data_dir):
    data_path = data_dir / "trajectories.npy"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = np.load(data_path, mmap_mode="r")

    if data.ndim != 4:
        raise ValueError(
            f"Expected trajectories with 4 dimensions, got shape {data.shape}"
        )

    if data.dtype != np.float32:
        raise ValueError(f"Expected float32 dataset, got {data.dtype}")

    train_ids = np.load(data_dir / "train_indices.npy")
    val_ids = np.load(data_dir / "val_indices.npy")
    test_ids = np.load(data_dir / "test_indices.npy")

    if data.shape[1] < 2:
        raise ValueError("Each trajectory must contain at least two snapshots")

    n = data.shape[-1]

    if data.shape[-2] != n:
        raise ValueError(f"Expected square spatial grids, got {data.shape[-2:]}")

    if 2 * MODES1 > n or MODES2 > n // 2 + 1:
        raise ValueError(
            f"Fourier modes ({MODES1}, {MODES2}) do not fit grid size {n}"
        )

    return data, train_ids, val_ids, test_ids


def make_pair_batch(data, trajectory_ids, pair_indices, n_transitions):
    local_sample_ids = pair_indices // n_transitions
    time_ids = pair_indices % n_transitions
    sample_ids = trajectory_ids[local_sample_ids]

    x = np.asarray(data[sample_ids, time_ids], dtype=np.float32)
    y = np.asarray(data[sample_ids, time_ids + 1], dtype=np.float32)

    return jnp.asarray(x), jnp.asarray(y)


def save_checkpoint(path, params, epoch, val_loss, history):
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "params": jax.device_get(params),
        "model_config": MODEL_CONFIG,
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "history": history,
    }

    temp_path = path.with_suffix(path.suffix + ".tmp")

    with open(temp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    temp_path.replace(path)


def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "params" in payload:
        config = payload.get("model_config")

        if config is not None and config != MODEL_CONFIG:
            raise ValueError(
                "Checkpoint architecture does not match this script.\n"
                f"Checkpoint: {config}\n"
                f"Current: {MODEL_CONFIG}"
            )

        params = payload["params"]
    else:
        params = payload
        payload = {"params": params, "epoch": None, "val_loss": None}

    params = jax.tree_util.tree_map(jnp.asarray, params)
    return params, payload


def evaluate_pairs(
    params, data, trajectory_ids, batch_size=BATCH_SIZE, pair_indices=None
):
    n_transitions = data.shape[1] - 1
    total_pairs = len(trajectory_ids) * n_transitions

    if pair_indices is None:
        pair_indices = np.arange(total_pairs, dtype=np.int64)

    errors = []

    for i in range(0, len(pair_indices), batch_size):
        batch_ids = pair_indices[i:i + batch_size]
        x, y = make_pair_batch(data, trajectory_ids, batch_ids, n_transitions)
        e = eval_step(params, x, y)
        errors.append(np.asarray(jax.device_get(e)))

    return float(np.concatenate(errors).mean())


@partial(jax.jit, static_argnames=("n_steps",))
def rollout(params, omega0, n_steps):
    def body(omega, _):
        next_omega = fno(params, omega[None, ...])[0]
        return next_omega, next_omega

    _, future = lax.scan(body, omega0, None, length=n_steps)
    return jnp.concatenate([omega0[None, ...], future], axis=0)


def evaluate_rollouts(params, data, trajectory_ids, n_samples=1):
    if len(trajectory_ids) == 0:
        return None

    selected = trajectory_ids[:min(n_samples, len(trajectory_ids))]
    final_errors = []
    mean_errors = []

    for sample_id in selected:
        truth = np.asarray(data[sample_id], dtype=np.float32)
        pred = rollout(
            params,
            jnp.asarray(truth[0]),
            n_steps=truth.shape[0] - 1,
        )
        pred = np.asarray(jax.device_get(pred), dtype=np.float32)

        num = np.linalg.norm(pred - truth, axis=(1, 2))
        den = np.linalg.norm(truth, axis=(1, 2)) + 1e-8
        errors = num / den

        mean_errors.append(float(errors[1:].mean()))
        final_errors.append(float(errors[-1]))

    return {
        "mean_rel_l2": float(np.mean(mean_errors)),
        "final_rel_l2": float(np.mean(final_errors)),
    }


def train(data, train_ids, val_ids, args):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")

    checkpoint_path = args.checkpoint
    latest_path = checkpoint_path.with_name(
        checkpoint_path.stem + "_latest" + checkpoint_path.suffix
    )
    history_path = checkpoint_path.with_name("history.json")
    n_transitions = data.shape[1] - 1
    n_train_pairs = len(train_ids) * n_transitions
    n_val_pairs = len(val_ids) * n_transitions

    if n_train_pairs == 0 or n_val_pairs == 0:
        raise ValueError("Training and validation splits must not be empty")

    pairs_per_epoch = min(args.pairs_per_epoch, n_train_pairs)
    val_pairs = min(args.val_pairs, n_val_pairs)
    steps_per_epoch = int(np.ceil(pairs_per_epoch / args.batch_size))
    total_steps = args.epochs * steps_per_epoch

    schedule = optax.cosine_decay_schedule(
        args.lr,
        total_steps,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.adamw(schedule, weight_decay=WEIGHT_DECAY),
    )

    params = init_fno(jax.random.PRNGKey(args.seed))
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(p, state, x, y):
        def loss_fn(current_params):
            pred = fno(current_params, x)
            return rel_l2(pred, y).mean()

        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)

        return p, state, loss

    rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng(args.seed + 1)
    val_pair_indices = val_rng.choice(
        n_val_pairs,
        size=val_pairs,
        replace=False,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rollout_mean_rel_l2": [],
        "val_rollout_final_rel_l2": [],
    }

    best_val = np.inf
    start_time = time.perf_counter()

    print("\n" + "=" * 65)
    print("2D NAVIER-STOKES FNO TRAINING")
    print("=" * 65)
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    print("Dataset shape:", data.shape)
    print("Grid:", data.shape[-2:])
    print("Train trajectories:", len(train_ids))
    print("Validation trajectories:", len(val_ids))
    print("Pairs per epoch:", pairs_per_epoch)
    print("Batch size:", args.batch_size)
    print("Steps per epoch:", steps_per_epoch)
    print("Total planned optimizer steps:", total_steps)
    print("Model parameters:", sum(x.size for x in jax.tree_util.tree_leaves(params)))
    print("=" * 65 + "\n")

    for ep in range(args.epochs):
        epoch_start = time.perf_counter()
        pair_indices = rng.choice(
            n_train_pairs,
            size=pairs_per_epoch,
            replace=False,
        )

        train_losses = []

        for i in range(0, pairs_per_epoch, args.batch_size):
            batch_ids = pair_indices[i:i + args.batch_size]
            x, y = make_pair_batch(
                data,
                train_ids,
                batch_ids,
                n_transitions,
            )

            params, opt_state, loss = train_step(params, opt_state, x, y)
            loss.block_until_ready()
            train_losses.append(float(loss))

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate_pairs(
            params,
            data,
            val_ids,
            batch_size=args.batch_size,
            pair_indices=val_pair_indices,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        is_best = val_loss < best_val

        if is_best:
            best_val = val_loss
            save_checkpoint(
                checkpoint_path,
                params,
                ep + 1,
                best_val,
                history,
            )

        save_checkpoint(
            latest_path,
            params,
            ep + 1,
            val_loss,
            history,
        )

        rollout_stats = None

        if (ep + 1) % ROLLOUT_CHECK_EVERY == 0:
            rollout_stats = evaluate_rollouts(
                params,
                data,
                val_ids,
                n_samples=ROLLOUT_VAL_SAMPLES,
            )

        if rollout_stats is None:
            history["val_rollout_mean_rel_l2"].append(None)
            history["val_rollout_final_rel_l2"].append(None)
        else:
            history["val_rollout_mean_rel_l2"].append(
                rollout_stats["mean_rel_l2"]
            )
            history["val_rollout_final_rel_l2"].append(
                rollout_stats["final_rel_l2"]
            )

        lr_now = float(schedule(jnp.asarray(ep * steps_per_epoch)))
        elapsed = time.perf_counter() - epoch_start
        status = " | BEST" if is_best else ""
        rollout_text = ""

        if rollout_stats is not None:
            rollout_text = (
                f" | rollout mean={rollout_stats['mean_rel_l2']:.4e}"
                f" final={rollout_stats['final_rel_l2']:.4e}"
            )

        print(
            f"Epoch {ep + 1:03d}/{args.epochs}"
            f" | train={train_loss:.6e}"
            f" | val={val_loss:.6e}"
            f" | lr={lr_now:.2e}"
            f" | time={elapsed:.1f}s"
            f"{rollout_text}{status}",
            flush=True,
        )

    total_time = time.perf_counter() - start_time

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    params, checkpoint = load_checkpoint(checkpoint_path)

    print("\nTraining complete.")
    print("Best epoch:", checkpoint["epoch"])
    print("Best validation loss:", checkpoint["val_loss"])
    print("Training time:", round(total_time, 2), "sec")
    print("Best checkpoint:", checkpoint_path)
    print("Latest checkpoint:", latest_path)

    return params


def run_trajectory(
    params, data, split_ids, split_name, trajectory, output_dir, data_dir
):
    if trajectory < 0 or trajectory >= len(split_ids):
        raise IndexError(
            f"Trajectory index {trajectory} is out of range for split "
            f"'{split_name}' with {len(split_ids)} trajectories"
        )

    sample_id = int(split_ids[trajectory])
    truth = np.asarray(data[sample_id], dtype=np.float32)
    n_steps = truth.shape[0] - 1
    omega0 = jnp.asarray(truth[0])

    start = time.perf_counter()
    pred = rollout(params, omega0, n_steps=n_steps)
    pred.block_until_ready()
    compile_time = time.perf_counter() - start

    start = time.perf_counter()
    pred = rollout(params, omega0, n_steps=n_steps)
    pred.block_until_ready()
    infer_time = time.perf_counter() - start

    pred = np.asarray(jax.device_get(pred), dtype=np.float32)

    num = np.linalg.norm(pred - truth, axis=(1, 2))
    den = np.linalg.norm(truth, axis=(1, 2)) + 1e-8
    errors = num / den

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{split_name}_{trajectory:03d}_sample_{sample_id:03d}"

    np.save(output_dir / f"{prefix}_prediction.npy", pred)
    np.save(output_dir / f"{prefix}_reference.npy", truth)
    np.save(output_dir / f"{prefix}_errors.npy", errors.astype(np.float32))

    times_path = data_dir / "times.npy"

    if times_path.exists():
        times = np.load(times_path)
    else:
        times = np.arange(truth.shape[0], dtype=np.float32) * 0.01

    np.save(output_dir / f"{prefix}_times.npy", times)

    print("\nAutoregressive rollout complete.")
    print("Split:", split_name)
    print("Trajectory index in split:", trajectory)
    print("Dataset sample id:", sample_id)
    print("Prediction shape:", pred.shape)
    print("Compilation + first run:", round(compile_time, 3), "sec")
    print("Rollout time after JIT:", round(infer_time, 4), "sec")
    print("Mean relative L2:", float(errors[1:].mean()))
    print("Final relative L2:", float(errors[-1]))
    print("Maximum relative L2:", float(errors.max()))
    print("Saved to:", output_dir)


def evaluate_model(params, data, val_ids, test_ids, batch_size):
    print("\nEvaluating one-step prediction...")

    val_loss = evaluate_pairs(
        params,
        data,
        val_ids,
        batch_size=batch_size,
    )

    test_loss = evaluate_pairs(
        params,
        data,
        test_ids,
        batch_size=batch_size,
    )

    print("Validation one-step relative L2:", val_loss)
    print("Test one-step relative L2:", test_loss)

    print("\nEvaluating autoregressive rollouts...")

    val_rollout = evaluate_rollouts(params, data, val_ids, n_samples=3)
    test_rollout = evaluate_rollouts(params, data, test_ids, n_samples=3)

    print("Validation rollout:", val_rollout)
    print("Test rollout:", test_rollout)

    results = {
        "val_one_step_rel_l2": val_loss,
        "test_one_step_rel_l2": test_loss,
        "val_rollout": val_rollout,
        "test_rollout": test_rollout,
    }

    with open(BASE_DIR / "evaluation.json", "w") as f:
        json.dump(results, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        nargs="?",
        choices=("train", "evaluate", "run"),
        default="train",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--pairs-per-epoch", type=int, default=PAIRS_PER_EPOCH)
    parser.add_argument("--val-pairs", type=int, default=VAL_PAIRS_PER_EPOCH)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "predictions")

    return parser.parse_args()


def main():
    args = parse_args()

    data, train_ids, val_ids, test_ids = load_dataset(args.data_dir)

    if args.mode == "train":
        params = train(data, train_ids, val_ids, args)
        evaluate_model(params, data, val_ids, test_ids, args.batch_size)

    else:
        params, checkpoint = load_checkpoint(args.checkpoint)
        print("Loaded checkpoint:", args.checkpoint)
        print("Checkpoint epoch:", checkpoint.get("epoch"))
        print("Checkpoint validation loss:", checkpoint.get("val_loss"))

        if args.mode == "evaluate":
            evaluate_model(params, data, val_ids, test_ids, args.batch_size)

        elif args.mode == "run":
            split_ids = {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            }[args.split]

            run_trajectory(
                params,
                data,
                split_ids,
                args.split,
                args.trajectory,
                args.output_dir,
                args.data_dir,
            )


if __name__ == "__main__":
    main()
