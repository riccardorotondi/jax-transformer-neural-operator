import jax
import jax.numpy as jnp


def init_mlp(key, layer_sizes):
    """Initialize weights and biases as a PyTree."""
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []

    for k, in_dim, out_dim in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        w = jax.random.normal(k, (in_dim, out_dim))
        w = w * jnp.sqrt(2.0 / in_dim)
        b = jnp.zeros((out_dim,))
        params.append({"w": w, "b": b})

    return params


def mlp_forward(params, x):
    """x shape: (..., input_dim)"""
    for layer in params[:-1]:
        x = jax.nn.gelu(x @ layer["w"] + layer["b"])

    last = params[-1]
    return x @ last["w"] + last["b"]


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    params = init_mlp(key, [2, 64, 64, 1])

    x = jnp.ones((8, 32, 2))
    y = mlp_forward(params, x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
