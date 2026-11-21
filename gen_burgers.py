import sys
import time
import jax.numpy as jnp
from jax import lax
import numpy as np

def grf(n_samples, N, rng, alpha=2.0, tau=5.0, sigma=25.0):
    k = np.fft.rfftfreq(N, d=1.0 / N)  # целые волновые числа 0..N/2
    # спектр поля: амплитуда падает с ростом k, среднее значение обнуляем
    coef = sigma * ((2 * np.pi * k) ** 2 + tau ** 2) ** (-alpha / 2)
    coef[0] = 0.0
    # случайные комплексные коэффициенты Фурье
    xi = rng.standard_normal((n_samples, len(k))) + 1j * rng.standard_normal((n_samples, len(k)))
    u_hat = coef * xi * N / jnp.sqrt(2)
    return np.fft.irfft(u_hat, n=N)

def make_solver(N, nu=0.01, T=1.0, dt=1e-4):

    # Волновые числа
    k = 2 * jnp.pi * jnp.fft.rfftfreq(N, d=1.0 / N)

    # Линейная часть уравнения
    L = -nu * k**2

    # Dealiasing
    dealias = (
        jnp.abs(jnp.fft.rfftfreq(N, d=1.0 / N)) < (N // 3)
    )

    # Интегрирующий множитель
    E = jnp.exp(L * dt)
    E2 = jnp.exp(L * dt / 2)

    # Нелинейная часть
    def Nl(vh):
        u = jnp.fft.irfft(vh * dealias, n=N, axis=-1)
        return -0.5j * k * jnp.fft.rfft(u * u, axis=-1)

    n_steps = int(round(T / dt))

    # Один временной шаг
    def step(i, vh):

        a = Nl(vh)

        b = Nl(E2 * (vh + dt / 2 * a))

        c = Nl(E2 * vh + dt / 2 * b)

        d = Nl(E * vh + dt * E2 * c)

        vh_new = E * vh + dt / 6 * (
            E * a + 2 * E2 * (b + c) + d
        )

        return vh_new

    # Полный решатель
    @jax.jit
    def solve_batch(u0):

        vh = jnp.fft.rfft(u0, axis=-1)

        vh = lax.fori_loop(
            0,
            n_steps,
            step,
            vh
        )

        uT = jnp.fft.irfft(vh, n=N, axis=-1)

        return uT

    return solve_batch

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, n = 1024, int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    u0 = grf(n, N, rng)
    t = time.time()
    uT = jnp.concatenate([solve(u0[i:i + 200]) for i in range(0, n, 200)])
    print("batch solve time", time.time() - t, "max|u0|", jnp.abs(u0).max(), "nan?", jnp.isnan(uT).any())
    # время одного решения нужно для сравнения с FNO
    t = time.time(); solve(u0[:1]); t1 = time.time() - t
    print("single-sample solver time (N=1024):", t1)
    np.savez_compressed("/home/void/NO/burgers.npz", u0=u0, uT=uT, solver_time=t1)
