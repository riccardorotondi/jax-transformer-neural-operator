import jax
import jax.numpy as jnp
import optax

from data import make_dataset
from model import init_mlp, mlp_forward


def relative_l2_loss(params, x, y):
    pred = mlp_forward(params, x)

    numerator = jnp.linalg.norm(pred - y, axis=(1, 2))
    denominator = jnp.linalg.norm(y, axis=(1, 2))

    return jnp.mean(numerator / (denominator + 1e-8))


def main():
    train_x, train_y, val_x, val_y = make_dataset(
        n_train=2048,
        n_val=256,
        n_points=32,
    )

    key = jax.random.PRNGKey(42)
    key_init, key_shuffle = jax.random.split(key)

    params = init_mlp(key_init, [2, 64, 64, 1])

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(relative_l2_loss)(params, x, y)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    batch_size = 64
    num_epochs = 100

    for epoch in range(num_epochs):
        permutation = jax.random.permutation(
            jax.random.fold_in(key_shuffle, epoch),
            train_x.shape[0],
        )

        train_x_shuffled = train_x[permutation]
        train_y_shuffled = train_y[permutation]

        losses = []

        for start in range(0, train_x.shape[0], batch_size):
            xb = train_x_shuffled[start : start + batch_size]
            yb = train_y_shuffled[start : start + batch_size]

            params, opt_state, loss = train_step(params, opt_state, xb, yb)
            losses.append(loss)

        if epoch % 10 == 0:
            val_loss = relative_l2_loss(params, val_x, val_y)

            print(
                f"Epoch {epoch:03d} | "
                f"Train: {jnp.mean(jnp.stack(losses)):.4f} | "
                f"Val: {val_loss:.4f}"
            )


if __name__ == "__main__":
    main()
