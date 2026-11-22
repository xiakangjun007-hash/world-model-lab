import unittest
import numpy as np

from world_model import DT, experiment, make_trajectories, rollout, simulator_step


class WorldModelTests(unittest.TestCase):
    def test_force_changes_motion(self):
        state = np.zeros(2)
        np.testing.assert_allclose(simulator_step(state, 1.0), [DT**2, DT])
        np.testing.assert_allclose(simulator_step(state, -1.0), [-DT**2, -DT])

    def test_rollout_matches_dataset(self):
        states, actions = make_trajectories(np.random.default_rng(5), 4, 13)
        np.testing.assert_allclose(rollout(simulator_step, states[:, 0], actions), states)

    def test_model_beats_baselines(self):
        scores, truth, pred = experiment(epochs=35)
        for baseline in ("freeze_state", "ignore_action"):
            self.assertTrue(np.all(scores["world_model"] < scores[baseline]), scores)
        self.assertLess(truth[0], truth[1])
        self.assertLess(pred[0], pred[1])


if __name__ == "__main__":
    unittest.main()
