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


class GeoTransformer(nn.Module):
    """Detector-Transformer that is told where each detector is: the (x, y, t) coordinates stim attaches to every
    detector are normalised and projected into the embedding (instead of a learned 1-D position the model must discover),
    and a learned [CLS] token, not a mean over all tokens, carries the decision (review of 4 October 2026)."""
    coords: tuple            # ((x, y, t), ...) one per detector, from stim's get_detector_coordinates()
    width: int = 64
    depth: int = 2
    heads: int = 4

    @nn.compact
    def __call__(self, x):
        c = jnp.asarray(self.coords, dtype=jnp.float32)
        c = (c - c.mean(axis=0)) / (c.std(axis=0) + 1e-6)
        h = nn.Embed(2, self.width)(x.astype(jnp.int32)) + nn.Dense(self.width)(c)
        cls = self.param("cls", nn.initializers.normal(0.02), (1, 1, self.width))
        h = jnp.concatenate([jnp.broadcast_to(cls, (h.shape[0], 1, self.width)), h], axis=1)
        for _ in range(self.depth):
            h = h + nn.SelfAttention(num_heads=self.heads)(nn.LayerNorm()(h))
            h = h + nn.Dense(self.width)(nn.gelu(nn.Dense(4 * self.width)(nn.LayerNorm()(h))))
        return nn.Dense(1)(nn.LayerNorm()(h[:, 0]))[..., 0]


def detector_coords(circuit) -> tuple:
    """((x, y, t), ...) for every detector of a stim circuit, in detector order; padded to 3 numbers."""
    co = circuit.get_detector_coordinates()
    return tuple(tuple((list(co[i]) + [0.0, 0.0, 0.0])[:3]) for i in range(len(co)))


@dataclass
class TrainResult:
    test_error: float
    train_seconds: float
    steps: int
    seconds_per_step: float
    best_step: int = -1          # with validation: the step whose parameters were kept (-1: the last step, no validation)
    predictions: object = None   # the kept model's 0/1 prediction for every test shot (for paired tests)


def train(model: nn.Module, x_tr, y_tr, x_te, y_te, *, steps: int, batch: int, lr: float = 1e-3, seed: int = 0,
          val_frac: float = 0.0, eval_every: int = 100) -> TrainResult:
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
    n_val = int(len(x_tr) * val_frac)
    if n_val:                                                       # the LAST val_frac of the training shots, never trained on
        x_va, y_va = x_tr[-n_val:], y_tr[-n_val:]
        x_tr, y_tr = x_tr[:-n_val], y_tr[:-n_val]
        vloss = jax.jit(loss_fn)

        def val_loss(p):
            return float(np.mean([float(vloss(p, jnp.asarray(x_va[i:i + batch]), jnp.asarray(y_va[i:i + batch])))
                                  for i in range(0, len(x_va), batch)]))
    n = len(x_tr)
    xb0, yb0 = jnp.asarray(x_tr[:batch]), jnp.asarray(y_tr[:batch])
    params_c, state_c, _ = step(params, state, xb0, yb0)          # compile outside the timed loop
    jax.block_until_ready(params_c)
    best, best_loss, best_step = params, float("inf"), -1
    t0 = time.perf_counter()
    for k in range(1, steps + 1):
        idx = rng.integers(0, n, batch)
        params, state, _ = step(params, state, jnp.asarray(x_tr[idx]), jnp.asarray(y_tr[idx]))
        if n_val and (k % eval_every == 0 or k == steps):
            vl = val_loss(params)
            if vl < best_loss:
                best, best_loss, best_step = params, vl, k
    jax.block_until_ready(params)
    dt = time.perf_counter() - t0                                   # with validation, includes the validation passes
    if n_val:
        params = best
    ev = jax.jit(model.apply)                                       # evaluation in chunks of the training batch: a fixed
    pred = np.concatenate([np.asarray(ev(params, jnp.asarray(x_te[i:i + batch])) > 0)   # chunk of 4096 made the attention of
                           for i in range(0, len(x_te), batch)]).astype(np.uint8)      # 336 detectors allocate 7.4 GB
    return TrainResult(float(np.mean(pred != y_te)), dt, steps, dt / steps, best_step, pred)
