"""Functional training utilities for JAX + Equinox models."""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from hipposvgp.models.svgp import OnlineSVGP
from hipposvgp.models.sgpr import OnlineSGPR
from hipposvgp.types import SpectralSample


def make_svgp_train_step(
    optimizer: optax.GradientTransformation,
    filter_spec: Any = None,
):
    """Create a JIT-compiled SVGP training step.

    Args:
        optimizer: optax optimizer
        filter_spec: pytree of booleans specifying which parameters to train.
                     If None, all parameters are trained.

    Returns:
        A function (model, opt_state, X, y, sample, key) -> (model, opt_state, loss, aux)
    """

    @eqx.filter_jit
    def step(
        model: OnlineSVGP,
        opt_state: Any,
        X: Float[Array, "N D"],
        y: Float[Array, "N 1"],
        sample: SpectralSample | None,
        key: PRNGKeyArray,
    ):
        if filter_spec is not None:
            trainable, frozen = eqx.partition(model, filter_spec)

            def loss_fn(trainable_):
                m = eqx.combine(trainable_, frozen)
                elbo, ve, kl, km = m.elbo(X, y, sample=sample, key=key)
                return -elbo, (ve, kl, km)

            (loss, aux), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=True
            )(trainable)
            updates, new_opt_state = optimizer.update(grads, opt_state, trainable)
            new_trainable = eqx.apply_updates(trainable, updates)
            new_model = eqx.combine(new_trainable, frozen)
        else:

            def loss_fn(m):
                elbo, ve, kl, km = m.elbo(X, y, sample=sample, key=key)
                return -elbo, (ve, kl, km)

            (loss, aux), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=True
            )(model)
            updates, new_opt_state = optimizer.update(grads, opt_state, model)
            new_model = eqx.apply_updates(model, updates)

        return new_model, new_opt_state, loss, aux

    return step


def make_sgpr_train_step(
    optimizer: optax.GradientTransformation,
    filter_spec: Any = None,
):
    """Create a JIT-compiled SGPR training step."""

    @eqx.filter_jit
    def step(
        model: OnlineSGPR,
        opt_state: Any,
        sample: SpectralSample | None,
        key: PRNGKeyArray | None,
    ):
        if filter_spec is not None:
            trainable, frozen = eqx.partition(model, filter_spec)

            def loss_fn(trainable_):
                m = eqx.combine(trainable_, frozen)
                elbo, km, resolved_sample = m.elbo(sample=sample, key=key)
                return -elbo, (km, resolved_sample)

            (loss, aux), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=True
            )(trainable)
            updates, new_opt_state = optimizer.update(grads, opt_state, trainable)
            new_trainable = eqx.apply_updates(trainable, updates)
            new_model = eqx.combine(new_trainable, frozen)
        else:

            def loss_fn(m):
                elbo, km, resolved_sample = m.elbo(sample=sample, key=key)
                return -elbo, (km, resolved_sample)

            (loss, aux), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=True
            )(model)
            updates, new_opt_state = optimizer.update(grads, opt_state, model)
            new_model = eqx.apply_updates(model, updates)

        return new_model, new_opt_state, loss, aux

    return step


def make_exact_gp_train_step(
    optimizer: optax.GradientTransformation,
):
    """Create a JIT-compiled ExactGP training step."""
    from hipposvgp.models.exact import ExactGP

    @eqx.filter_jit
    def step(
        model: ExactGP,
        opt_state: Any,
        X: Float[Array, "N D"],
        y: Float[Array, "N P"],
    ):
        def loss_fn(m):
            return m.nll(X, y)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    return step


def train_loop(
    model: Any,
    num_epochs: int,
    step_fn: Callable,
    step_args_fn: Callable,
    *,
    key: PRNGKeyArray | None = None,
    log_every: int = 1000,
    verbose: bool = False,
) -> tuple[Any, list[dict]]:
    """Generic training loop.

    Args:
        model: Equinox model (returned as updated)
        num_epochs: number of training epochs
        step_fn: function (model, opt_state, *args) -> (model, opt_state, loss, ...)
        step_args_fn: function (epoch, key) -> tuple of extra args for step_fn
        key: PRNG key (split per epoch)
        log_every: print loss every N epochs
        verbose: whether to print progress

    Returns:
        (model, history): updated model and list of loss dicts
    """
    history = []

    for epoch in range(num_epochs):
        if key is not None:
            key, subkey = jax.random.split(key)
        else:
            subkey = None

        args = step_args_fn(epoch, subkey)
        result = step_fn(*args)
        model = result[0]
        loss = result[2]

        loss_val = float(loss)
        record = {"epoch": epoch, "loss": loss_val}
        history.append(record)

        if verbose and ((epoch + 1) % log_every == 0 or epoch == 0):
            print(f"  epoch {epoch + 1:5d}  loss={loss_val:.4f}")

    return model, history
