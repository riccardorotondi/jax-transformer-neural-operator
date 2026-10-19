# JAX Transformer Neural Operator

A minimal implementation of a Transformer-based Neural Operator using JAX.

## Goal

Learn the solution operator of the 1D Poisson equation:

\[
-u''(x) = f(x), \quad x \in (0,1)
\]

with homogeneous Dirichlet boundary conditions.

The model learns a mapping between functions:

\[
\mathcal{G}: f(x) \mapsto u(x)
\]

## Tech Stack

- JAX
- Optax
- NumPy
- Matplotlib

## Planned Architecture

- Coordinate and function-value embeddings
- Multi-head self-attention
- Transformer encoder blocks
- Pointwise output projection

## Experiments

- Training on a 32-point spatial grid
- Evaluation on unseen forcing functions
- Resolution transfer from 32 to 64 points
- Comparison with a simple MLP baseline

## Status

Work in progress — interview preparation project.
