"""Learned decoders in JAX/Flax: an MLP over the flattened detector record, and a small Transformer that reads each
detector as a token (the family of recurrent/attention decoders used for surface-code decoding). Both output one logit:
did the logical observable flip."""
from __future__ import annotations

import time
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


class MLP(nn.Module):
    hidden: int = 256

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(x)[..., 0]


class DetectorTransformer(nn.Module):
    n_det: int
    width: int = 64
    depth: int = 2
    heads: int = 4

    @nn.compact
    def __call__(self, x):
        tok = nn.Embed(2, self.width)(x.astype(jnp.int32))                        # detector value 0/1
        pos = self.param("pos", nn.initializers.normal(0.02), (self.n_det, self.width))
        h = tok + pos
        for _ in range(self.depth):
            a = nn.SelfAttention(num_heads=self.heads)(nn.LayerNorm()(h))
            h = h + a
            h = h + nn.Dense(self.width)(nn.gelu(nn.Dense(4 * self.width)(nn.LayerNorm()(h))))
        return nn.Dense(1)(jnp.mean(nn.LayerNorm()(h), axis=-2))[..., 0]


@dataclass
class TrainResult:
    test_error: float
    train_seconds: float
    steps: int
    seconds_per_step: float


def train(model: nn.Module, x_tr, y_tr, x_te, y_te, *, steps: int, batch: int, lr: float = 1e-3, seed: int = 0) -> TrainResult:
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.asarray(x_tr[:2]))
    opt = optax.adam(lr)
    state = opt.init(params)

    def loss_fn(p, xb, yb):
        return optax.sigmoid_binary_cross_entropy(model.apply(p, xb), yb.astype(jnp.float32)).mean()

    @jax.jit
    def step(p, s, xb, yb):
        l, g = jax.value_and_grad(loss_fn)(p, xb, yb)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l

    rng = np.random.default_rng(seed)
    n = len(x_tr)
    xb0, yb0 = jnp.asarray(x_tr[:batch]), jnp.asarray(y_tr[:batch])
    params_c, state_c, _ = step(params, state, xb0, yb0)          # compile outside the timed loop
    jax.block_until_ready(params_c)
    t0 = time.perf_counter()
    for _ in range(steps):
        idx = rng.integers(0, n, batch)
        params, state, _ = step(params, state, jnp.asarray(x_tr[idx]), jnp.asarray(y_tr[idx]))
    jax.block_until_ready(params)
    dt = time.perf_counter() - t0
    ev = jax.jit(model.apply)                                       # evaluation in chunks of the training batch: a fixed
    pred = np.concatenate([np.asarray(ev(params, jnp.asarray(x_te[i:i + batch])) > 0)   # chunk of 4096 made the attention of
                           for i in range(0, len(x_te), batch)]).astype(np.uint8)      # 336 detectors allocate 7.4 GB
    return TrainResult(float(np.mean(pred != y_te)), dt, steps, dt / steps)
