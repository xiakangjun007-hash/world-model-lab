"""A small action-conditioned world model for a damped oscillator."""

import argparse
import numpy as np

DT = 0.05


def simulator_step(state, action):
    """One semi-implicit Euler step. state[..., :] = (position, velocity)."""
    x, v = np.moveaxis(np.asarray(state), -1, 0)
    acceleration = -1.4 * x - 0.3 * v - 0.12 * x**3 + np.asarray(action)
    v_next = v + DT * acceleration
    return np.stack((x + DT * v_next, v_next), axis=-1)


def make_trajectories(rng, count, steps, interval=5):
    """Sample independent trajectories, with piecewise-constant inputs."""
    actions = rng.uniform(-1.2, 1.2, (count, (steps + interval - 1) // interval))
    actions = np.repeat(actions, interval, axis=1)[:, :steps]
    states = np.empty((count, steps + 1, 2))
    states[:, 0, :] = rng.uniform(-1.35, 1.35, (count, 2))
    for t in range(steps):
        states[:, t + 1] = simulator_step(states[:, t], actions[:, t])
    return states, actions


class WorldModel:
    """Predict a state increment with a tiny, manually differentiated MLP."""

    def __init__(self, rng, hidden=32):
        self.w1 = rng.normal(0, 0.5, (3, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 0.1, (hidden, 2))
        self.b2 = np.zeros(2)
        self.delta_scale = np.ones(2)

    def fit(self, states, actions, rng, epochs=90, batch_size=256, lr=0.03):
        x = np.column_stack((states[:, :-1].reshape(-1, 2), actions.ravel()))
        delta = np.diff(states, axis=1).reshape(-1, 2)
        self.delta_scale = np.maximum(delta.std(axis=0), 1e-8)
        y = delta / self.delta_scale
        for _ in range(epochs):
            indices = rng.permutation(len(x))
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                features = x[batch]
                hidden = np.tanh(features @ self.w1 + self.b1)
                error = hidden @ self.w2 + self.b2 - y[batch]
                d_out = 2 * error / error.size
                d_hidden = (d_out @ self.w2.T) * (1 - hidden**2)
                self.w2 -= lr * (hidden.T @ d_out)
                self.b2 -= lr * d_out.sum(axis=0)
                self.w1 -= lr * (features.T @ d_hidden)
                self.b1 -= lr * d_hidden.sum(axis=0)

    def step(self, state, action):
        state = np.asarray(state, dtype=float)
        a = np.broadcast_to(action, state.shape[:-1])
        features = np.concatenate((state, a[..., None]), axis=-1)
        hidden = np.tanh(features @ self.w1 + self.b1)
        return state + (hidden @ self.w2 + self.b2) * self.delta_scale


def rollout(step, initial, actions):
    """Roll out without feeding any future true states back into the model."""
    initial = np.asarray(initial)
    actions = np.asarray(actions)
    if initial.shape[-1:] != (2,) or initial.shape[:-1] != actions.shape[:-1]:
        raise ValueError("initial and action batch dimensions must match")
    states = [initial]
    for a in np.moveaxis(actions, -1, 0):
        states.append(step(states[-1], a))
    return np.stack(states, axis=-2)


def rmse_by_coordinate(pred, truth):
    return np.sqrt(np.mean((pred[..., 1:, :] - truth[..., 1:, :])**2,
                           axis=tuple(range(pred.ndim - 1))))


def experiment(seed=7, epochs=90):
    # Separate streams prevent the test set from changing with training settings.
    train_rng, test_rng, model_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    )
    train_s, train_a = make_trajectories(train_rng, 192, 50)
    test_s, test_a = make_trajectories(test_rng, 48, 80, interval=8)
    model = WorldModel(model_rng)
    model.fit(train_s, train_a, model_rng, epochs=epochs)
    predicted = rollout(model.step, test_s[:, 0], test_a)
    frozen = rollout(lambda s, a: s, test_s[:, 0], test_a)
    ignored = rollout(model.step, test_s[:, 0], np.zeros_like(test_a))

    x0 = np.array([0.3, 0.0])
    left, right = (np.full(40, u) for u in (-0.8, 0.8))
    true_end = [rollout(simulator_step, x0, a)[-1, 0] for a in (left, right)]
    pred_end = [rollout(model.step, x0, a)[-1, 0] for a in (left, right)]
    return {
        "world_model": rmse_by_coordinate(predicted, test_s),
        "freeze_state": rmse_by_coordinate(frozen, test_s),
        "ignore_action": rmse_by_coordinate(ignored, test_s),
    }, true_end, pred_end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=90)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    scores, true_end, pred_end = experiment(args.seed, args.epochs)
    print("Held-out 80-step rollout RMSE (lower is better)")
    for name, (x, v) in scores.items():
        print(f"{name:15s} position={x:.4f}  velocity={v:.4f}")
    print("Same initial state, force -0.8 vs +0.8, final position:")
    print(f"  simulator: {true_end[0]:+.3f}, {true_end[1]:+.3f}")
    print(f"  learned:   {pred_end[0]:+.3f}, {pred_end[1]:+.3f}")


if __name__ == "__main__":
    main()
