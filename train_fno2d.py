
import json
import math
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("dataset_ns_256")
OUTPUT_DIR = Path("fno_ns_results_256")

SEED = 42

WIDTH = 32
MODES = 24
LAYERS = 4

BATCH_SIZE = 2
TEST_BATCH_SIZE = 2

EPOCHS = 100
STEPS_PER_EPOCH = 300
PATIENCE = 15

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_FRACTION = 0.05

BETA1 = 0.9
BETA2 = 0.999
ADAM_EPS = 1e-8

MAX_GRAD_NORM = 1.0

ROLLOUT_HORIZONS = [1, 10, 25, 50, 100]


# ============================================================
# FNO MODEL
# ============================================================

def init_dense(key, in_features, out_features):
    limit = math.sqrt(6.0 / (in_features + out_features))

    weight = jax.random.uniform(
        key,
        (in_features, out_features),
        minval=-limit,
        maxval=limit
    )

    bias = jnp.zeros((out_features,), dtype=jnp.float32)

    return weight, bias


def init_complex(key, shape, scale):
    key_real, key_imag = jax.random.split(key)

    real = jax.random.normal(key_real, shape)
    imag = jax.random.normal(key_imag, shape)

    return ((real + 1j * imag) * scale).astype(jnp.complex64)


def init_fno(key):
    keys = jax.random.split(key, 2 + LAYERS * 4 + 4)
    index = 0

    # Vorticity + x + y coordinates
    lift_w, lift_b = init_dense(keys[index], 3, WIDTH)
    index += 1

    layers = []

    for _ in range(LAYERS):
        w_pos = init_complex(
            keys[index],
            (MODES, MODES, WIDTH, WIDTH),
            1.0 / (WIDTH * WIDTH)
        )
        index += 1

        w_neg = init_complex(
            keys[index],
            (MODES, MODES, WIDTH, WIDTH),
            1.0 / (WIDTH * WIDTH)
        )
        index += 1

        local_w, local_b = init_dense(
            keys[index],
            WIDTH,
            WIDTH
        )
        index += 1

        layers.append({
            "w_pos": w_pos,
            "w_neg": w_neg,
            "local_w": local_w,
            "local_b": local_b
        })

    proj1_w, proj1_b = init_dense(
        keys[index],
        WIDTH,
        WIDTH * 2
    )
    index += 1

    # Initially predict zero increment
    proj2_w = jnp.zeros(
        (WIDTH * 2, 1),
        dtype=jnp.float32
    )
    proj2_b = jnp.zeros((1,), dtype=jnp.float32)

    return {
        "lift_w": lift_w,
        "lift_b": lift_b,
        "layers": layers,
        "proj1_w": proj1_w,
        "proj1_b": proj1_b,
        "proj2_w": proj2_w,
        "proj2_b": proj2_b
    }


def dense(x, weight, bias):
    return jnp.einsum("...i,io->...o", x, weight) + bias


def spectral_conv(x, w_pos, w_neg):
    batch, nx, ny, channels = x.shape

    x_ft = jnp.fft.rfft2(
        x,
        axes=(1, 2)
    )

    out_ft = jnp.zeros(
        (batch, nx, ny // 2 + 1, channels),
        dtype=jnp.complex64
    )

    positive = jnp.einsum(
        "bxyi,xyio->bxyo",
        x_ft[:, :MODES, :MODES, :],
        w_pos
    )

    negative = jnp.einsum(
        "bxyi,xyio->bxyo",
        x_ft[:, -MODES:, :MODES, :],
        w_neg
    )

    out_ft = out_ft.at[:, :MODES, :MODES, :].set(positive)
    out_ft = out_ft.at[:, -MODES:, :MODES, :].set(negative)

    return jnp.fft.irfft2(
        out_ft,
        s=(nx, ny),
        axes=(1, 2)
    )


def make_grid(nx, ny):
    grid_x = jnp.linspace(
        0.0, 1.0, nx,
        endpoint=False,
        dtype=jnp.float32
    )

    grid_y = jnp.linspace(
        0.0, 1.0, ny,
        endpoint=False,
        dtype=jnp.float32
    )

    gx, gy = jnp.meshgrid(
        grid_x,
        grid_y,
        indexing="ij"
    )

    return jnp.stack([gx, gy], axis=-1)


def fno_apply(params, x):
    batch, nx, ny, channels = x.shape

    grid = make_grid(nx, ny)

    grid = jnp.broadcast_to(
        grid[None, ...],
        (batch, nx, ny, 2)
    )

    # Add spatial coordinates
    x = jnp.concatenate(
        [x, grid],
        axis=-1
    )

    # Lifting
    x = dense(
        x,
        params["lift_w"],
        params["lift_b"]
    )

    # Residual Fourier blocks
    for layer in params["layers"]:
        spectral = spectral_conv(
            x,
            layer["w_pos"],
            layer["w_neg"]
        )

        local = dense(
            x,
            layer["local_w"],
            layer["local_b"]
        )

        update = jax.nn.gelu(spectral + local)

        x = x + 0.1 * update

    # Projection
    x = jax.nn.gelu(
        dense(
            x,
            params["proj1_w"],
            params["proj1_b"]
        )
    )

    # Predict normalized increment
    delta = dense(
        x,
        params["proj2_w"],
        params["proj2_b"]
    )

    return delta


def count_parameters(params):
    total = 0

    for leaf in jax.tree_util.tree_leaves(params):
        multiplier = 2 if jnp.iscomplexobj(leaf) else 1
        total += leaf.size * multiplier

    return int(total)


# ============================================================
# LOSS
# ============================================================

def relative_l2(prediction, target):
    error = prediction - target

    error_norm = jnp.linalg.norm(
        error.reshape(error.shape[0], -1),
        axis=1
    )

    target_norm = jnp.linalg.norm(
        target.reshape(target.shape[0], -1),
        axis=1
    )

    return error_norm / jnp.maximum(target_norm, 1e-12)


# ============================================================
# OPTAX TRAINING
# ============================================================

def make_train_step(optimizer):

    @jax.jit
    def train_step(
        params,
        optimizer_state,
        x,
        y,
        data_mean,
        data_std
    ):
        x_norm = (x - data_mean) / data_std
        y_norm = (y - data_mean) / data_std

        def loss_function(current_params):
            delta = fno_apply(current_params, x_norm)

            prediction = x_norm + delta

            return jnp.mean(
                relative_l2(prediction, y_norm)
            )

        loss, grads = jax.value_and_grad(loss_function)(params)

        grad_norm = optax.global_norm(grads)

        updates, optimizer_state = optimizer.update(
            grads,
            optimizer_state,
            params
        )

        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss, grad_norm

    return train_step


@jax.jit
def validation_step(params, x, y, data_mean, data_std):
    x_norm = (x - data_mean) / data_std
    y_norm = (y - data_mean) / data_std

    prediction = x_norm + fno_apply(params, x_norm)

    return jnp.mean(relative_l2(prediction, y_norm))


@jax.jit
def predict_physical(params, x, data_mean, data_std):
    x_norm = (x - data_mean) / data_std

    delta = fno_apply(params, x_norm)

    prediction_norm = x_norm + delta

    return prediction_norm * data_std + data_mean


@partial(jax.jit, static_argnames=("n_steps",))
def autoregressive_rollout(
    params,
    initial,
    data_mean,
    data_std,
    n_steps
):
    state = (initial - data_mean) / data_std

    def body(current_state, _):
        delta = fno_apply(params, current_state)

        next_state = current_state + delta

        return next_state, next_state

    _, predictions = lax.scan(
        body,
        state,
        None,
        length=n_steps
    )

    return predictions * data_std + data_mean


# ============================================================
# DATA HELPERS
# ============================================================

def make_pairs(sample_ids, n_transitions):
    return np.asarray(
        [
            (int(sample_id), time_id)
            for sample_id in sample_ids
            for time_id in range(n_transitions)
        ],
        dtype=np.int32
    )


def get_batch(data, pairs):
    x = np.asarray(
        data[pairs[:, 0], pairs[:, 1]],
        dtype=np.float32
    )

    y = np.asarray(
        data[pairs[:, 0], pairs[:, 1] + 1],
        dtype=np.float32
    )

    return (
        jnp.asarray(x[..., None]),
        jnp.asarray(y[..., None])
    )


def calculate_normalization(data, train_ids):
    total_sum = 0.0
    total_squared_sum = 0.0
    total_count = 0

    for sample_id in train_ids:
        trajectory = np.asarray(
            data[int(sample_id)],
            dtype=np.float64
        )

        total_sum += float(np.sum(trajectory))
        total_squared_sum += float(np.sum(trajectory ** 2))
        total_count += trajectory.size

    data_mean = total_sum / total_count

    variance = (
        total_squared_sum / total_count
        - data_mean ** 2
    )

    data_std = math.sqrt(max(variance, 1e-12))

    return np.float32(data_mean), np.float32(data_std)


def save_params(params, path):
    leaves, _ = jax.tree_util.tree_flatten(params)

    arrays = {
        f"arr_{index}": np.asarray(jax.device_get(value))
        for index, value in enumerate(leaves)
    }

    np.savez(path, **arrays)


def evaluate_validation(
    params,
    data,
    pairs,
    data_mean,
    data_std
):
    total_loss = 0.0
    total_samples = 0

    for start in range(0, len(pairs), BATCH_SIZE):
        batch_pairs = pairs[start:start + BATCH_SIZE]

        x, y = get_batch(data, batch_pairs)

        loss = validation_step(
            params,
            x,
            y,
            data_mean,
            data_std
        )

        batch_count = len(batch_pairs)

        total_loss += float(jax.device_get(loss)) * batch_count
        total_samples += batch_count

    return total_loss / total_samples


# ============================================================
# ONE-STEP TEST
# ============================================================

def test_one_step(
    params,
    data,
    pairs,
    data_mean,
    data_std
):
    relative_errors = []
    baseline_relative_errors = []

    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    baseline_squared_error_sum = 0.0

    element_count = 0

    for start in range(0, len(pairs), TEST_BATCH_SIZE):
        batch_pairs = pairs[start:start + TEST_BATCH_SIZE]

        x, y = get_batch(data, batch_pairs)

        prediction = predict_physical(
            params,
            x,
            data_mean,
            data_std
        )

        prediction = np.asarray(
            jax.device_get(prediction)
        )[..., 0]

        x_np = np.asarray(
            jax.device_get(x)
        )[..., 0]

        y_np = np.asarray(
            jax.device_get(y)
        )[..., 0]

        error = prediction - y_np
        baseline_error = x_np - y_np

        pred_norm = np.linalg.norm(
            error.reshape(len(batch_pairs), -1),
            axis=1
        )

        target_norm = np.linalg.norm(
            y_np.reshape(len(batch_pairs), -1),
            axis=1
        )

        base_norm = np.linalg.norm(
            baseline_error.reshape(len(batch_pairs), -1),
            axis=1
        )

        denominator = np.maximum(target_norm, 1e-12)

        relative_errors.extend(
            (pred_norm / denominator).tolist()
        )

        baseline_relative_errors.extend(
            (base_norm / denominator).tolist()
        )

        squared_error_sum += float(np.sum(error ** 2))
        absolute_error_sum += float(np.sum(np.abs(error)))

        baseline_squared_error_sum += float(
            np.sum(baseline_error ** 2)
        )

        element_count += error.size

    return {
        "pairs": int(len(relative_errors)),
        "rmse": float(
            math.sqrt(squared_error_sum / element_count)
        ),
        "mae": float(
            absolute_error_sum / element_count
        ),
        "relative_l2_mean": float(
            np.mean(relative_errors)
        ),
        "relative_l2_std": float(
            np.std(relative_errors)
        ),
        "persistence_baseline_rmse": float(
            math.sqrt(baseline_squared_error_sum / element_count)
        ),
        "persistence_baseline_relative_l2_mean": float(
            np.mean(baseline_relative_errors)
        ),
        "relative_l2_ratio_to_baseline": float(
            np.mean(relative_errors)
            / max(np.mean(baseline_relative_errors), 1e-12)
        )
    }


# ============================================================
# AUTOREGRESSIVE ROLLOUT TEST
# ============================================================

def test_rollout(
    params,
    data,
    test_ids,
    data_mean,
    data_std,
    n_steps
):
    model_errors = []
    baseline_errors = []

    for sample_id in test_ids:
        initial_np = np.asarray(
            data[int(sample_id), 0],
            dtype=np.float32
        )

        true_trajectory = np.asarray(
            data[int(sample_id), 1:n_steps + 1],
            dtype=np.float32
        )

        initial = jnp.asarray(
            initial_np[None, :, :, None]
        )

        prediction = autoregressive_rollout(
            params,
            initial,
            data_mean,
            data_std,
            n_steps=n_steps
        )

        prediction = np.asarray(
            jax.device_get(prediction)
        )[:, 0, :, :, 0]

        model_difference = prediction - true_trajectory

        baseline_prediction = np.broadcast_to(
            initial_np,
            true_trajectory.shape
        )

        baseline_difference = (
            baseline_prediction - true_trajectory
        )

        target_norm = np.linalg.norm(
            true_trajectory.reshape(n_steps, -1),
            axis=1
        )

        denominator = np.maximum(target_norm, 1e-12)

        model_norm = np.linalg.norm(
            model_difference.reshape(n_steps, -1),
            axis=1
        )

        baseline_norm = np.linalg.norm(
            baseline_difference.reshape(n_steps, -1),
            axis=1
        )

        model_errors.append(model_norm / denominator)
        baseline_errors.append(baseline_norm / denominator)

    model_errors = np.asarray(model_errors, dtype=np.float64)
    baseline_errors = np.asarray(baseline_errors, dtype=np.float64)

    model_mean = np.mean(model_errors, axis=0)
    model_std = np.std(model_errors, axis=0)

    baseline_mean = np.mean(baseline_errors, axis=0)
    baseline_std = np.std(baseline_errors, axis=0)

    selected_horizons = {}

    for horizon in ROLLOUT_HORIZONS:
        index = horizon - 1

        selected_horizons[str(horizon)] = {
            "fno_relative_l2_mean": float(model_mean[index]),
            "fno_relative_l2_std": float(model_std[index]),
            "persistence_relative_l2_mean": float(
                baseline_mean[index]
            ),
            "persistence_relative_l2_std": float(
                baseline_std[index]
            )
        }

    return {
        "trajectories": int(len(test_ids)),
        "steps": int(n_steps),
        "selected_horizons": selected_horizons,
        "fno_relative_l2_mean_by_step": model_mean.tolist(),
        "fno_relative_l2_std_by_step": model_std.tolist(),
        "persistence_relative_l2_mean_by_step": baseline_mean.tolist(),
        "persistence_relative_l2_std_by_step": baseline_std.tolist(),
        "final_horizon_relative_l2_mean": float(model_mean[-1]),
        "final_horizon_relative_l2_std": float(model_std[-1]),
        "final_horizon_persistence_relative_l2_mean": float(
            baseline_mean[-1]
        )
    }, model_errors, baseline_errors


# ============================================================
# MAIN
# ============================================================

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    data_path = DATA_DIR / "trajectories.npy"
    train_path = DATA_DIR / "train_indices.npy"
    val_path = DATA_DIR / "val_indices.npy"
    test_path = DATA_DIR / "test_indices.npy"

    for path in [data_path, train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset file not found: {path}"
            )

    data = np.load(data_path, mmap_mode="r")

    if data.ndim != 4:
        raise ValueError(
            f"Expected 4D dataset, received shape {data.shape}"
        )

    n_samples, n_snapshots, nx, ny = data.shape

    if nx != 256 or ny != 256:
        raise ValueError(
            f"Expected 256x256 dataset, received {nx}x{ny}"
        )

    if 2 * MODES >= nx or MODES > ny // 2 + 1:
        raise ValueError(
            "MODES is too large for dataset resolution"
        )

    if n_snapshots - 1 < max(ROLLOUT_HORIZONS):
        raise ValueError(
            "Not enough snapshots for requested rollout horizons"
        )

    train_ids = np.load(train_path).astype(np.int32)
    val_ids = np.load(val_path).astype(np.int32)
    test_ids = np.load(test_path).astype(np.int32)

    all_ids = np.concatenate([
        train_ids,
        val_ids,
        test_ids
    ])

    if len(np.unique(all_ids)) != n_samples:
        raise ValueError(
            "Dataset splits overlap or do not cover all trajectories"
        )

    if len(all_ids) != n_samples:
        raise ValueError(
            "Dataset split sizes do not match number of samples"
        )

    if (
        len(train_ids) == 0
        or len(val_ids) == 0
        or len(test_ids) == 0
    ):
        raise ValueError("Train, validation and test splits must be nonempty")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("FNO-2D TRAINING AND TESTING")
    print("=" * 65)

    print("Backend:", jax.default_backend())
    print("Devices:", jax.devices())
    print("Dataset shape:", data.shape)

    print("Train trajectories:", len(train_ids))
    print("Validation trajectories:", len(val_ids))
    print("Test trajectories:", len(test_ids))

    print("Resolution:", f"{nx}x{ny}")
    print("FNO width:", WIDTH)
    print("FNO modes:", MODES)
    print("FNO layers:", LAYERS)
    print("Batch size:", BATCH_SIZE)

    if jax.default_backend() != "gpu":
        print("WARNING: JAX is not using a GPU backend.")

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    data_mean_np, data_std_np = calculate_normalization(
        data,
        train_ids
    )

    data_mean = jnp.asarray(
        data_mean_np,
        dtype=jnp.float32
    )

    data_std = jnp.asarray(
        data_std_np,
        dtype=jnp.float32
    )

    print("Data mean:", float(data_mean_np))
    print("Data std:", float(data_std_np))

    # --------------------------------------------------------
    # DATA PAIRS
    # --------------------------------------------------------

    n_transitions = n_snapshots - 1

    train_pairs = make_pairs(train_ids, n_transitions)
    val_pairs = make_pairs(val_ids, n_transitions)
    test_pairs = make_pairs(test_ids, n_transitions)

    rng = np.random.default_rng(SEED)

    steps_per_epoch = min(
        STEPS_PER_EPOCH,
        len(train_pairs) // BATCH_SIZE
    )

    if steps_per_epoch < 1:
        raise ValueError("Not enough training pairs for one batch")

    total_steps = EPOCHS * steps_per_epoch

    print("\nTrain pairs:", len(train_pairs))
    print("Validation pairs:", len(val_pairs))
    print("Test pairs:", len(test_pairs))
    print("Steps per epoch:", steps_per_epoch)
    print("Total planned optimizer steps:", total_steps)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    key = jax.random.PRNGKey(SEED)

    params = init_fno(key)

    nparams = count_parameters(params)

    print("\nModel parameters:", f"{nparams:,}")

    # --------------------------------------------------------
    # OPTAX
    # --------------------------------------------------------

    warmup_steps = max(
        1,
        int(total_steps * WARMUP_FRACTION)
    )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=LEARNING_RATE,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=3e-5
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adamw(
            learning_rate=schedule,
            b1=BETA1,
            b2=BETA2,
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY
        )
    )

    optimizer_state = optimizer.init(params)

    train_step = make_train_step(optimizer)

    best_params = params
    best_val_loss = float("inf")
    best_epoch = 0

    patience_counter = 0
    history = []

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print("\nStarting training...\n")

    training_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()

        selected_count = steps_per_epoch * BATCH_SIZE

        selected_indices = rng.choice(
            len(train_pairs),
            size=selected_count,
            replace=False
        )

        epoch_pairs = train_pairs[selected_indices]

        losses = []
        grad_norms = []

        for step_index in range(steps_per_epoch):
            start = step_index * BATCH_SIZE

            batch_pairs = epoch_pairs[
                start:start + BATCH_SIZE
            ]

            x, y = get_batch(data, batch_pairs)

            params, optimizer_state, loss, grad_norm = train_step(
                params,
                optimizer_state,
                x,
                y,
                data_mean,
                data_std
            )

            losses.append(loss)
            grad_norms.append(grad_norm)

        train_loss = float(
            jax.device_get(jnp.mean(jnp.stack(losses)))
        )

        mean_grad_norm = float(
            jax.device_get(jnp.mean(jnp.stack(grad_norms)))
        )

        val_loss = evaluate_validation(
            params,
            data,
            val_pairs,
            data_mean,
            data_std
        )

        epoch_elapsed = time.perf_counter() - epoch_start

        current_lr = float(
            schedule((epoch - 1) * steps_per_epoch)
        )

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_params = params
            patience_counter = 0

            save_params(
                best_params,
                OUTPUT_DIR / "best_fno_weights.npz"
            )
        else:
            patience_counter += 1

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "learning_rate": current_lr,
            "mean_gradient_norm": mean_grad_norm,
            "seconds": epoch_elapsed
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train={train_loss:.6e} | "
            f"val={val_loss:.6e} | "
            f"lr={current_lr:.2e} | "
            f"grad={mean_grad_norm:.3e} | "
            f"time={epoch_elapsed:.1f}s | "
            f"{'BEST' if improved else 'patience ' + str(patience_counter)}",
            flush=True
        )

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    jax.block_until_ready(best_params)

    training_seconds = time.perf_counter() - training_start

    print("\nTraining complete.")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_val_loss)
    print("Training time:", round(training_seconds, 2), "sec")

    # --------------------------------------------------------
    # ONE-STEP TEST
    # --------------------------------------------------------

    print("\nRunning one-step test on all test pairs...")

    one_step_results = test_one_step(
        best_params,
        data,
        test_pairs,
        data_mean,
        data_std
    )

    print(
        "One-step relative L2:",
        one_step_results["relative_l2_mean"]
    )

    print(
        "Persistence baseline:",
        one_step_results["persistence_baseline_relative_l2_mean"]
    )

    print("One-step RMSE:", one_step_results["rmse"])
    print("One-step MAE:", one_step_results["mae"])

    print(
        "Relative L2 / baseline:",
        one_step_results["relative_l2_ratio_to_baseline"]
    )

    # --------------------------------------------------------
    # ROLLOUT
    # --------------------------------------------------------

    print("\nRunning autoregressive rollout test...")

    rollout_results, rollout_errors, baseline_errors = test_rollout(
        best_params,
        data,
        test_ids,
        data_mean,
        data_std,
        n_transitions
    )

    for horizon, values in rollout_results["selected_horizons"].items():
        print(
            f"Horizon {int(horizon):3d} | "
            f"FNO rel L2={values['fno_relative_l2_mean']:.6e} | "
            f"Persistence={values['persistence_relative_l2_mean']:.6e}"
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results = {
        "model": {
            "name": "FNO-2D-Residual",
            "resolution": [nx, ny],
            "width": WIDTH,
            "modes": MODES,
            "layers": LAYERS,
            "nparams": nparams,
            "input": "normalized vorticity + x,y coordinates",
            "target": "normalized vorticity increment",
            "normalization_mean": float(data_mean_np),
            "normalization_std": float(data_std_np),
            "loss": "mean relative L2"
        },
        "training": {
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "training_seconds": training_seconds,
            "batch_size": BATCH_SIZE,
            "steps_per_epoch": steps_per_epoch,
            "learning_rate_initial": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "train_trajectories": int(len(train_ids)),
            "validation_trajectories": int(len(val_ids)),
            "test_trajectories": int(len(test_ids)),
            "history": history
        },
        "one_step_test": one_step_results,
        "autoregressive_rollout_test": rollout_results
    }

    with open(OUTPUT_DIR / "test_results.json", "w") as file:
        json.dump(results, file, indent=4)

    np.savez(
        OUTPUT_DIR / "rollout_errors.npz",
        fno=rollout_errors,
        persistence=baseline_errors
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    try:
        import matplotlib.pyplot as plt

        steps = np.arange(1, n_transitions + 1)

        fno_mean = np.asarray(
            rollout_results["fno_relative_l2_mean_by_step"]
        )

        fno_std = np.asarray(
            rollout_results["fno_relative_l2_std_by_step"]
        )

        base_mean = np.asarray(
            rollout_results["persistence_relative_l2_mean_by_step"]
        )

        plt.figure(figsize=(10, 5))

        plt.plot(
            steps,
            fno_mean,
            label="FNO autoregressive"
        )

        plt.fill_between(
            steps,
            np.maximum(fno_mean - fno_std, 0.0),
            fno_mean + fno_std,
            alpha=0.2
        )

        plt.plot(
            steps,
            base_mean,
            label="Persistence baseline"
        )

        plt.xlabel("Prediction horizon")
        plt.ylabel("Relative L2 error")
        plt.title("FNO-2D autoregressive test")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "rollout_error.png",
            dpi=150
        )

        plt.close()

        epochs = [item["epoch"] for item in history]
        train_losses = [item["train_loss"] for item in history]
        val_losses = [item["validation_loss"] for item in history]

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label="Train")
        plt.plot(epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Relative L2")
        plt.title("FNO-2D training history")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "training_history.png",
            dpi=150
        )

        plt.close()

    except ImportError:
        print("Matplotlib not installed; plots skipped.")

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    config = {
        "data_dir": str(DATA_DIR),
        "output_dir": str(OUTPUT_DIR),
        "seed": SEED,
        "resolution": [nx, ny],
        "width": WIDTH,
        "modes": MODES,
        "layers": LAYERS,
        "nparams": nparams,
        "batch_size": BATCH_SIZE,
        "epochs_limit": EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_fraction": WARMUP_FRACTION,
        "patience": PATIENCE,
        "loss": "mean relative L2",
        "dataset_shape": list(data.shape),
        "device_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()]
    }

    with open(OUTPUT_DIR / "training_config.json", "w") as file:
        json.dump(config, file, indent=4)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FNO TRAINING AND TESTING COMPLETE")
    print("=" * 65)

    print("Parameters:", f"{nparams:,}")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_val_loss)
    print("Training time:", round(training_seconds, 2), "sec")

    print(
        "One-step relative L2:",
        one_step_results["relative_l2_mean"]
    )

    print(
        "Rollout final relative L2:",
        rollout_results["final_horizon_relative_l2_mean"]
    )

    print("Results directory:", OUTPUT_DIR)
    print("Saved: best_fno_weights.npz")
    print("Saved: test_results.json")
    print("Saved: training_config.json")
    print("Saved: rollout_errors.npz")
    print("Saved: rollout_error.png")
    print("Saved: training_history.png")


if __name__ == "__main__":
    main()
