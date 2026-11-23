import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


N = 128
nu = 1e-3
n_steps = 1000


def init_omega(key, omega_rms=1.0, N=128, K0=8.0):
    k = jnp.fft.fftfreq(N, d=1 / N)

    noise = jax.random.normal(
        key,
        (N, N),
        dtype=jnp.float32
    )

    kx, ky = jnp.meshgrid(
        k,
        k,
        indexing="ij"
    )

    k_squared = kx**2 + ky**2

    filter_ = jnp.exp(
        -k_squared / (2 * K0**2)
    )

    noise_hat = jnp.fft.fft2(noise)

    omega_hat = noise_hat * filter_
    omega_hat = omega_hat.at[0, 0].set(0.0)

    omega = jnp.fft.ifft2(omega_hat).real

    rms = jnp.sqrt(jnp.mean(omega**2))
    omega = omega * (omega_rms / rms)

    return omega.astype(jnp.float32)


def get_velocity(omega_hat, kx, ky, k_squared):
    psi_hat = jnp.where(
        k_squared > 0,
        omega_hat / jnp.where(k_squared > 0, k_squared, 1.0),
        0.0
    )

    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat

    u = jnp.fft.ifft2(u_hat).real
    v = jnp.fft.ifft2(v_hat).real

    return u, v


def navier_stokes_rhs(
    omega_hat,
    kx,
    ky,
    k_squared,
    dealias,
    nu
):
    u, v = get_velocity(
        omega_hat,
        kx,
        ky,
        k_squared
    )

    omega_x_hat = 1j * kx * omega_hat
    omega_y_hat = 1j * ky * omega_hat

    omega_x = jnp.fft.ifft2(omega_x_hat).real
    omega_y = jnp.fft.ifft2(omega_y_hat).real

    advection = u * omega_x + v * omega_y

    advection_hat = jnp.fft.fft2(advection)
    advection_hat = advection_hat * dealias

    diffusion_hat = -nu * k_squared * omega_hat

    rhs_hat = -advection_hat + diffusion_hat

    return rhs_hat


def rk4_step(omega_hat, dt, kx, ky, k_squared, dealias, nu):
    def rhs(w):
        return navier_stokes_rhs(
            w,
            kx,
            ky,
            k_squared,
            dealias,
            nu
        )

    k1 = rhs(omega_hat)

    k2 = rhs(
        omega_hat + 0.5 * dt * k1
    )

    k3 = rhs(
        omega_hat + 0.5 * dt * k2
    )

    k4 = rhs(
        omega_hat + dt * k3
    )

    return omega_hat + (dt / 6.0) * (
        k1 + 2 * k2 + 2 * k3 + k4
    )


def simulate(omega0, dt, n_steps, kx, ky, k_squared, dealias, nu):
    omega_hat0 = jnp.fft.fft2(omega0)

    def body(omega_hat, _):
        omega_hat_next = rk4_step(
            omega_hat,
            dt,
            kx,
            ky,
            k_squared,
            dealias,
            nu
        )

        return omega_hat_next, omega_hat_next

    final_hat, trajectory_hat = lax.scan(
        body,
        omega_hat0,
        None,
        length=n_steps
    )

    trajectory = jnp.fft.ifft2(
        trajectory_hat,
        axes=(-2, -1)
    ).real

    return trajectory


def simulate_final(
    omega0,
    dt,
    n_steps,
    kx,
    ky,
    k_squared,
    dealias,
    nu
):
    omega_hat0 = jnp.fft.fft2(omega0)

    def body(omega_hat, _):
        omega_hat_next = rk4_step(
            omega_hat,
            dt,
            kx,
            ky,
            k_squared,
            dealias,
            nu
        )

        return omega_hat_next, None

    final_hat, _ = lax.scan(
        body,
        omega_hat0,
        None,
        length=n_steps
    )

    return jnp.fft.ifft2(final_hat).real


def make_grid(N):
    k = jnp.fft.fftfreq(N, d=1 / N)

    kx, ky = jnp.meshgrid(
        k,
        k,
        indexing="ij"
    )

    k_squared = kx**2 + ky**2

    dealias = (
        (jnp.abs(kx) <= N / 3) &
        (jnp.abs(ky) <= N / 3)
    ).astype(jnp.float32)

    return kx, ky, k_squared, dealias


N_DATA = N

NU_DATA = nu

DT_DATA = 0.001
T_DATA = 1.0

N_STEPS_DATA = round(T_DATA / DT_DATA)

SAVE_EVERY = 10

N_SAVED_STEPS = N_STEPS_DATA // SAVE_EVERY

N_SNAPSHOTS = N_SAVED_STEPS + 1

N_SAMPLES = 256

K0_DATA = 8.0

OMEGA_RMS = 1.0

ROOT_SEED = 20261123

OUTPUT_DIR = Path("dataset_ns_128")

assert N_STEPS_DATA % SAVE_EVERY == 0

assert N_DATA % 2 == 0

kx_data, ky_data, k2_data, dealias_data = make_grid(
    N_DATA
)


def generate_initial_condition(sample_key):
    omega = init_omega(
        sample_key,
        omega_rms=OMEGA_RMS,
        N=N_DATA,
        K0=K0_DATA
    )

    omega_hat = jnp.fft.fft2(omega)

    omega_hat = omega_hat * dealias_data

    omega = jnp.fft.ifft2(omega_hat).real

    omega = omega - jnp.mean(omega)

    rms = jnp.sqrt(jnp.mean(omega**2))

    omega = omega * (OMEGA_RMS / rms)

    return omega.astype(jnp.float32)


@partial(
    jax.jit,
    static_argnames=("save_every", "n_snapshots")
)
def simulate_sampled(
    omega0,
    dt,
    kx,
    ky,
    k_squared,
    dealias,
    nu,
    save_every,
    n_snapshots
):
    omega_hat0 = jnp.fft.fft2(omega0)

    def save_block(omega_hat, _):
        def integration_step(carry, _):
            next_hat = rk4_step(
                carry,
                dt,
                kx,
                ky,
                k_squared,
                dealias,
                nu
            )

            return next_hat, None

        omega_hat, _ = lax.scan(
            integration_step,
            omega_hat,
            None,
            length=save_every
        )

        snapshot = jnp.fft.ifft2(omega_hat).real

        return omega_hat, snapshot

    _, saved_fields = lax.scan(
        save_block,
        omega_hat0,
        None,
        length=n_snapshots - 1
    )

    trajectory = jnp.concatenate(
        [omega0[None, :, :], saved_fields],
        axis=0
    )

    return trajectory


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATA_PATH = OUTPUT_DIR / "trajectories.npy"

if DATA_PATH.exists():
    raise FileExistsError(
        f"{DATA_PATH} already exists. "
        "Choose another output directory or move the old dataset."
    )


dataset = np.lib.format.open_memmap(
    DATA_PATH,
    mode="w+",
    dtype=np.float32,
    shape=(
        N_SAMPLES,
        N_SNAPSHOTS,
        N_DATA,
        N_DATA
    )
)


root_key = jax.random.PRNGKey(ROOT_SEED)

print("\n" + "=" * 65)
print("NAVIER-STOKES DATASET GENERATION")
print("=" * 65)

print("JAX backend:", jax.default_backend())
print("JAX devices:", jax.devices())

print("Samples:", N_SAMPLES)
print("Resolution:", N_DATA)
print("Snapshots per trajectory:", N_SNAPSHOTS)
print("Integration steps:", N_STEPS_DATA)
print("Save interval:", SAVE_EVERY * DT_DATA)
print("Total simulated time:", T_DATA)

total_bytes = (
    N_SAMPLES
    * N_SNAPSHOTS
    * N_DATA
    * N_DATA
    * 4
)

print(
    "Expected dataset size:",
    round(total_bytes / (1024**3), 3),
    "GiB"
)


statistics = []

generation_start = time.perf_counter()

for sample_id in range(N_SAMPLES):
    sample_start = time.perf_counter()

    print(
        f"\nGenerating sample "
        f"{sample_id + 1}/{N_SAMPLES}"
    )

    sample_key = jax.random.fold_in(
        root_key,
        sample_id
    )

    omega0 = generate_initial_condition(
        sample_key
    )

    trajectory = simulate_sampled(
        omega0=omega0,
        dt=DT_DATA,
        kx=kx_data,
        ky=ky_data,
        k_squared=k2_data,
        dealias=dealias_data,
        nu=NU_DATA,
        save_every=SAVE_EVERY,
        n_snapshots=N_SNAPSHOTS
    )

    trajectory_np = np.asarray(
        jax.device_get(trajectory),
        dtype=np.float32
    )

    if not np.isfinite(trajectory_np).all():
        raise RuntimeError(
            f"NaN or Inf detected in sample {sample_id}"
        )

    means = trajectory_np.mean(axis=(1, 2))

    enstrophy = 0.5 * np.mean(
        trajectory_np**2,
        axis=(1, 2)
    )

    max_mean = np.max(np.abs(means))

    if max_mean > 1e-5:
        raise RuntimeError(
            f"Mean vorticity violation in sample {sample_id}"
        )

    if np.any(np.diff(enstrophy) > 1e-6):
        raise RuntimeError(
            f"Enstrophy increased in sample {sample_id}"
        )

    dataset[sample_id] = trajectory_np

    dataset.flush()

    stats = {
        "sample": sample_id,
        "initial_rms": float(
            np.sqrt(np.mean(trajectory_np[0]**2))
        ),
        "final_enstrophy": float(enstrophy[-1]),
        "final_mean": float(means[-1]),
        "max_abs_vorticity": float(
            np.max(np.abs(trajectory_np))
        )
    }

    statistics.append(stats)

    elapsed = time.perf_counter() - sample_start

    print(
        f"Initial RMS: {stats['initial_rms']:.6f}"
    )

    print(
        f"Final enstrophy: {stats['final_enstrophy']:.6f}"
    )

    print(
        f"Maximum mean: {max_mean:.3e}"
    )

    print(
        f"Generation time: {elapsed:.2f} sec"
    )

    print("Sample saved successfully.")

    del trajectory_np
    del trajectory
    del omega0


print("\nCreating dataset splits...")

rng = np.random.default_rng(ROOT_SEED)

indices = rng.permutation(N_SAMPLES)

n_train = int(0.8 * N_SAMPLES)
n_val = int(0.1 * N_SAMPLES)

train_indices = indices[:n_train]
val_indices = indices[n_train:n_train + n_val]
test_indices = indices[n_train + n_val:]


np.save(
    OUTPUT_DIR / "train_indices.npy",
    train_indices
)

np.save(
    OUTPUT_DIR / "val_indices.npy",
    val_indices
)

np.save(
    OUTPUT_DIR / "test_indices.npy",
    test_indices
)


times = (
    np.arange(N_SNAPSHOTS, dtype=np.float32)
    * SAVE_EVERY
    * DT_DATA
)

np.save(
    OUTPUT_DIR / "times.npy",
    times
)


metadata = {
    "equation": "2D incompressible Navier-Stokes",
    "formulation": "vorticity",
    "solver": "pseudo-spectral FFT + RK4",
    "resolution": N_DATA,
    "domain": [0.0, 2.0 * np.pi],
    "viscosity": NU_DATA,
    "dt_integrator": DT_DATA,
    "dt_saved": SAVE_EVERY * DT_DATA,
    "save_every": SAVE_EVERY,
    "simulation_time": T_DATA,
    "integration_steps": N_STEPS_DATA,
    "snapshots_per_trajectory": N_SNAPSHOTS,
    "samples": N_SAMPLES,
    "initial_condition": {
        "type": "Gaussian-filtered white noise",
        "K0": K0_DATA,
        "RMS": OMEGA_RMS,
        "zero_mean": True
    },
    "dtype": "float32",
    "array_shape": [
        N_SAMPLES,
        N_SNAPSHOTS,
        N_DATA,
        N_DATA
    ],
    "array_axes": [
        "sample",
        "time",
        "x",
        "y"
    ],
    "random_seed": ROOT_SEED,
    "dealiasing": "2/3 rule"
}


with open(
    OUTPUT_DIR / "metadata.json",
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=4
    )


with open(
    OUTPUT_DIR / "statistics.json",
    "w"
) as f:
    json.dump(
        statistics,
        f,
        indent=4
    )


del dataset

total_elapsed = time.perf_counter() - generation_start

print("\n" + "=" * 65)
print("DATASET GENERATION COMPLETE")
print("=" * 65)

print("Output directory:", OUTPUT_DIR)

print("Trajectory shape:", (
    N_SAMPLES,
    N_SNAPSHOTS,
    N_DATA,
    N_DATA
))

print("Training samples:", len(train_indices))
print("Validation samples:", len(val_indices))
print("Test samples:", len(test_indices))

print(
    "Dataset size:",
    round(total_bytes / (1024**3), 3),
    "GiB"
)

print(
    "Total generation time:",
    round(total_elapsed, 2),
    "sec"
)

print("\nFiles:")

for path in OUTPUT_DIR.iterdir():
    print(" ", path.name)

print("\nREADY FOR FNO TRAINING!")
