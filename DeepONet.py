import numpy as np
from pathlib import Path
import optax
import jax
import jax.numpy as jnp

BASE_DIR = Path(__file__).resolve().parent

D = np.load(BASE_DIR / "burgers.npz")

u0, uT = D["u0"].astype(np.float32) , D["uT"].astype(np.float32)
NTR, NTE = 1000, 100
FULL = u0.shape[1]
P = 128 # размер латентного пространства
M = 256 # кол - во сенсоров

def with_grid(a, res):
    return a[:, :: FULL // res]

branch_input = jnp.asarray(with_grid(u0, 256))
trunk_input = jnp.asarray(np.linspace(0, 1, branch_input.shape[1], endpoint=False))
target = jnp.asarray(with_grid(uT, 256))

def init_mlp(key, sizes):
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for i in range(len(sizes) - 1):
        fan_in = sizes[i]
        fan_out = sizes[i+1]
        W = jax.random.normal(
            keys[i], (fan_in, fan_out)
        )*jnp.sqrt(2)
        b = jnp.zeros(fan_out,)
        params.append((W, b))
    return params

def brach_apply(params, x):
    for i, (W, b) in enumerate(params):
        x = x @ W +b
        if i!=len(params) - 1:
            x= jax.nn.relu(x)
    return x

branch_params = init_mlp(jax.random.PRNGKey(0), [256,256,256,128])

branch_out = brach_apply(branch_params, branch_input)

print(branch_out.shape)
