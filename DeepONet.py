from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

BASE_DIR = Path(__file__).resolve().parent

D = np.load(BASE_DIR / "burgers.npz")

u0, uT = D["u0"].astype(np.float32) , D["uT"].astype(np.float32)
NTR, NTE = 1000, 100
FULL = u0.shape[1]
P = 128 # размер латентного пространства
M = 256 # кол - во сенсоров
EPOCHS = 1000

def with_grid(a, res):
    return a[:, :: FULL // res]

branch_input = jnp.asarray(with_grid(u0, 256))
trunk_input = jnp.asarray(np.linspace(0, 1, branch_input.shape[1], endpoint=False)).reshape(256,1)
target = jnp.asarray(with_grid(uT, 256))


# Обучающая выборка
branch_train = branch_input[:NTR]
target_train = target[:NTR]

# Тестовая выборка
branch_test = branch_input[NTR:NTR + NTE]
target_test = target[NTR:NTR + NTE]

def init_mlp(key, sizes):
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for i in range(len(sizes) - 1):
        fan_in = sizes[i]
        fan_out = sizes[i+1]
        W = jax.random.normal(
            keys[i], (fan_in, fan_out)
        )*jnp.sqrt(2.0 / fan_in)
        b = jnp.zeros((fan_out,))
        params.append((W, b))
    return params

def mlp_apply(params, x):
    for i, (W, b) in enumerate(params):
        x = x @ W +b
        if i!=len(params) - 1:
            x= jax.nn.relu(x)
    return x

def deeponet_apply(params, branch_input, trunk_input):
    branch_params, trunk_params = params

    branch_out = mlp_apply(branch_params, branch_input)
    trunk_out = mlp_apply(trunk_params, trunk_input)

    prediction = jnp.einsum("bp,xp->bx", branch_out, trunk_out)

    return prediction

def loss_fn(params, branch_input, trunk_input, target):
    prediction = deeponet_apply(params, branch_input, trunk_input)
    return jnp.mean((prediction - target)**2)

def train_step(params, opt_state, branch_input, trunk_input, target):

    loss, grads = jax.value_and_grad(loss_fn)(
        params,
        branch_input,
        trunk_input,
        target
    )

    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        params
    )

    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


branch_params = init_mlp(jax.random.PRNGKey(0), [256,256,256,128])
trunk_params = init_mlp(jax.random.PRNGKey(1), [1,128,128,128])
params = (branch_params, trunk_params)

optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

initial_loss = loss_fn(params, branch_input, trunk_input, target)

for epoch in range(EPOCHS):

    params, opt_state, loss = train_step(
        params,
        opt_state,
        branch_train,
        trunk_input,
        target_train
    )

    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")
new_loss = loss_fn(params, branch_input, trunk_input, target)

test_loss = loss_fn(
    params,
    branch_test,
    trunk_input,
    target_test
)

# 1024 координаты исходной сетки
x_1024 = np.linspace(
    0.0, 1.0, 1024,
    endpoint=False,
    dtype=np.float32
)

trunk_1024 = jnp.asarray(x_1024).reshape(1024, 1)

# Все тестовые примеры: те же индексы, что использовались при split
branch_test = branch_input[NTR:NTR + NTE]
target_test_1024 = jnp.asarray(
    uT[NTR:NTR + NTE],
    dtype=jnp.float32
)

# Предсказание для всей тестовой выборки
prediction_test_1024 = deeponet_apply(
    params,
    branch_test,
    trunk_1024
)

# Общий MSE
test_loss_1024 = jnp.mean(
    (prediction_test_1024 - target_test_1024) ** 2
)

print("Prediction shape:", prediction_test_1024.shape)
print("Target shape:", target_test_1024.shape)
print("Test MSE (1024 points):", float(test_loss_1024))

print("Test loss:", test_loss)

print("Initial loss:", initial_loss)
print("Loss during step:", loss)
print("New loss:", new_loss)
