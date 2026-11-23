import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import lax


N = 128
nu = 1e-3

key  = jax.random.PRNGKey(9)

def init_omega(key, omega_rms= 1.0, N = 128, K0 = 8.0):
    k  = jnp.fft.fftfreq(N, d = 1 / N)
    noise = jax.random.normal(key, (N, N), dtype = jnp.float32)
    kx, ky = jnp.meshgrid(k, k, indexing = "ij" )
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
        omega_hat, kx, ky, k_squared
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
            w, kx, ky, k_squared,
            dealias, nu
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
        k1 + 2*k2 + 2*k3 + k4
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

k = jnp.fft.fftfreq(N, d=1/N)

kx, ky = jnp.meshgrid(
    k, k, indexing="ij"
)

k_squared = kx**2 + ky**2

dealias = (
    (jnp.abs(kx) <= N/3) &
    (jnp.abs(ky) <= N/3)
).astype(jnp.float32)



omega = init_omega(key)

omega_hat = jnp.fft.fft2(omega)

power = jnp.abs(omega_hat)**2

power = jnp.fft.fftshift(power)

plt.figure(figsize=(8, 6))

plt.imshow(
    np.log10(np.asarray(power) + 1e-10),
    origin="lower",
    cmap="inferno"
)

plt.colorbar(label="log10(Power)")
plt.title("2D Fourier Power Spectrum")

plt.xlabel("kx")
plt.ylabel("ky")

plt.show()

print("Shape:", omega.shape)
print("Dtype:", omega.dtype)
print("Mean:", jnp.mean(omega))
print("RMS:", jnp.sqrt(jnp.mean(omega**2)))
