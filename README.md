# World Model Lab

Can a neural network learn to predict how a spring moves under an external force? This small, action-conditioned world model learns the dynamics of a damped oscillator from simulated trajectories. The model and backpropagation are written in NumPy, so you can read the entire training loop. No GPU, external dataset, or pretrained weights required.

## Run it

Use Python 3.10 or newer. From the repository directory:

```bash
python -m pip install -r requirements.txt
python world_model.py
python -m unittest -v
```

## Read the code in this order

1. **`simulator_step`: the environment.** The state is position and velocity; the action is an applied force. A semi-implicit Euler step advances the oscillator. A small cubic restoring term makes the dynamics nonlinear.
2. **`WorldModel`: the learned transition.** A 32-unit MLP takes position, velocity, and force and predicts the *change* in state. Training uses mean squared error on normalized state increments. The gradients in `fit` are written out using the chain rule.
3. **`rollout`: multi-step prediction.** After the initial state, the model feeds its own predictions back into the next step. It receives the future force sequence but never the future true states.
4. **`experiment`: evaluation.** The split holds out entire trajectories. Test trajectories are longer than training trajectories (80 vs. 50 steps), and their force switches at a different interval.

## Example results

One CPU run with seed 7 and 90 epochs. RMSE is calculated separately for each state component, excluding the known initial state. Lower is better.

| 80-step predictor | Position RMSE | Velocity RMSE |
| --- | ---: | ---: |
| Learned transition with the correct forces | 0.0803 | 0.0914 |
| Repeat the initial state | 1.0813 | 1.0267 |
| Same learned transition, but pass zero force | 0.3045 | 0.3616 |

Starting from the same state, constant forces of -0.8 and +0.8 give learned final positions of **-0.998** and **+0.650**, respectively, after 40 steps. The simulator gives **-0.973** and **+0.687**. Changing the action changes the predicted future.

Try `python world_model.py --epochs 10` to see how a shorter training run affects rollout accuracy.

## Scope and next step

This is an educational example with a fully observed state and a known future action sequence. It does not generate video, infer hidden state, or plan actions. The simulator creates the ground truth; the comparison is against simple baselines, not against a known-parameter solver. A useful next exercise is to observe only noisy positions, infer velocity from a short history, and compare the resulting multi-step predictions.
