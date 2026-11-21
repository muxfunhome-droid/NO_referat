import jax, jax.numpy as jnp, numpy as np, optax, time, json, pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

D = np.load(BASE_DIR / "burgers.npz")
u0, uT = D["u0"].astype(np.float32), D["uT"].astype(np.float32)
NTR, NTE = 1000, 100  # размеры обучающей и тестовой выборок
FULL = u0.shape[1]

def at(res, a):  # оставляет каждую (FULL // res)-ю точку, получается сетка из res точек
    return a[:, :: FULL // res]

def with_grid(a):
    # добавляем координату x вторым каналом
    x = jnp.linspace(0, 1, a.shape[1], endpoint=False)
    return jnp.stack([a, jnp.broadcast_to(x, a.shape)], -1)

WIDTH, MODES, LAYERS = 64, 16, 4

def init_fno(key):
    ks = jax.random.split(key, 4 + 2 * LAYERS)
    p = {"P": jax.random.normal(ks[0], (2, WIDTH)) / np.sqrt(2), "Pb": jnp.zeros(WIDTH),
         "Q1": jax.random.normal(ks[1], (WIDTH, 128)) / np.sqrt(WIDTH), "Q1b": jnp.zeros(128),
         "Q2": jax.random.normal(ks[2], (128, 1)) / np.sqrt(128), "Q2b": jnp.zeros(1), "L": []}
    # масштаб инициализации спектральных весов
    sc = 1.0 / (WIDTH * WIDTH)
    for l in range(LAYERS):
        kr, ki = jax.random.split(ks[4 + 2 * l])
        p["L"].append({"Rr": sc * jax.random.uniform(kr, (MODES, WIDTH, WIDTH)),
                       "Ri": sc * jax.random.uniform(ki, (MODES, WIDTH, WIDTH)),
                       "W": jax.random.normal(ks[5 + 2 * l], (WIDTH, WIDTH)) / np.sqrt(WIDTH),
                       "b": jnp.zeros(WIDTH)})
    return p

def fourier_layer(lp, v, last):
    n = v.shape[1]
    vh = jnp.fft.rfft(v, axis=1)                       # коэффициенты Фурье, форма (батч, n/2+1, каналы)
    R = lp["Rr"] + 1j * lp["Ri"]
    out_low = jnp.einsum("bkc,kcd->bkd", vh[:, :MODES], R)  # обучаемое умножение на R только для младших MODES мод
    out = jnp.zeros_like(vh).at[:, :MODES].set(out_low)    # остальные моды обнуляются
    k = jnp.fft.irfft(out, n=n, axis=1)
    # спектральная часть + локальное линейное преобразование W
    y = k + v @ lp["W"] + lp["b"]
    return y if last else jax.nn.gelu(y)

def fno(p, a):
    v = with_grid(a) @ p["P"] + p["Pb"]                # подъём в пространство каналов
    for l, lp in enumerate(p["L"]):
        v = fourier_layer(lp, v, l == LAYERS - 1)
    h = jax.nn.gelu(v @ p["Q1"] + p["Q1b"])            # проекция обратно в одну функцию
    return (h @ p["Q2"] + p["Q2b"])[..., 0]

# CNN для сравнения: фильтры заданы в точках сетки, поэтому она привязана к сетке обучения
CW, CK, CL = 64, 5, 7
DIL = [1, 2, 4, 8, 16, 32, 1]  # расширения свёрток подобраны так, чтобы поле зрения покрывало почти всю сетку из 256 точек
def init_cnn(key):
    ks = jax.random.split(key, CL)
    chans = [2] + [CW] * (CL - 1) + [1]
    return [{"w": jax.random.normal(ks[i], (CK, chans[i], chans[i + 1])) / np.sqrt(CK * chans[i]),
             "b": jnp.zeros(chans[i + 1])} for i in range(CL)]

def cnn(p, a):
    out = jnp.zeros(
        (v.shape[0], v.shape[1], lp["b"].shape[0]),
        dtype=v.dtype
    )

    for k in range(CK):
        xk = vp[:, k * DIL[i]:k * DIL[i] + v.shape[1], :]

        out = out + jnp.einsum(
            "bni,io->bno",
            xk,
            lp["w"][k]
        )

    v = out + lp["b"]
    return v[..., 0]

def rel_l2(pred, y):  # относительная ошибка в L2 по каждому образцу
    return jnp.linalg.norm(pred - y, axis=1) / jnp.linalg.norm(y, axis=1)

def train(model, params, res=256, epochs=300, bs=20, lr=1e-3, seed=0):
    xtr, ytr = jnp.array(at(res, u0[:NTR])), jnp.array(at(res, uT[:NTR]))
    # AdamW с косинусным убыванием шага
    steps = epochs * (NTR // bs)
    opt = optax.adamw(optax.cosine_decay_schedule(lr, steps), weight_decay=1e-4)
    st = opt.init(params)
    @jax.jit
    def step(params, st, x, y):
        l, g = jax.value_and_grad(lambda p: rel_l2(model(p, x), y).mean())(params)
        upd, st = opt.update(g, st, params)
        return optax.apply_updates(params, upd), st, l
    rng = np.random.default_rng(seed); hist = []
    for ep in range(epochs):
        idx = rng.permutation(NTR); ls = []
        for i in range(0, NTR, bs):
            params, st, l = step(
                params,
                st,
                xtr[idx[i:i + bs]],
                ytr[idx[i:i + bs]]
            )

            l.block_until_ready()
            ls.append(float(l))
        hist.append(float(np.mean(ls)))
        if ep % 50 == 0 or ep == epochs - 1:
            print(ep, hist[-1], flush=True)
    return params, hist

def evaluate(model, params, res):  # средняя ошибка на тестовых образцах, сетка res
    x, y = jnp.array(at(res, u0[NTR:NTR + NTE])), jnp.array(at(res, uT[NTR:NTR + NTE]))
    return float(rel_l2(model(params, x), y).mean())

if __name__ == "__main__":
    import os, sys
    out = json.load(open(BASE_DIR / "results.json")) if os.path.exists(BASE_DIR / "results.json") else {}
    # без аргументов обучаются обе модели, можно передать только FNO или только CNN
    todo = sys.argv[1:] or ["FNO", "CNN"]
    for name, model, init in [("FNO", fno, init_fno), ("CNN", cnn, init_cnn)]:
        if name not in todo: continue
        t = time.time()
        p, hist = train(model, init(jax.random.PRNGKey(0)))
        out[name] = {"train_time": time.time() - t, "hist": hist,
                     "test": {r: evaluate(model, p, r) for r in [64, 128, 256, 512, 1024]},
                     "nparams": int(sum(x.size for x in jax.tree_util.tree_leaves(p)))}
        print(name, out[name]["test"], out[name]["nparams"], out[name]["train_time"], flush=True)
        pickle.dump(jax.device_get(p), open(BASE_DIR / f"{name}.pkl", "wb"))
    # время одного прохода FNO на сетке 1024 (после прогрева jit), среднее по 100 запускам
    p = pickle.load(open(BASE_DIR / "FNO.pkl", "rb"))
    f = jax.jit(fno); x1 = jnp.array(u0[NTR:NTR + 1]); f(p, x1).block_until_ready()
    t = time.time()
    for _ in range(100):
        f(p, x1).block_until_ready()
    out["fno_infer_time_1024"] = (time.time() - t) / 100
    out["solver_time_1024"] = float(D["solver_time"])
    json.dump(out, open(BASE_DIR / "results.json", "w"), indent=1)
    print({k: v for k, v in out.items() if k not in ("FNO", "CNN")})
