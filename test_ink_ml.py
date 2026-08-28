import unittest

from ink_ml import InkDQN, InkGAN, InkRNN, dry_run, make_dataset, tokenize_source


class InkModelBehavior(unittest.TestCase):
    def test_given_a_graph_prompt_when_encoded_then_features_are_bounded(self):
        features = tokenize_source("input -> transform -> output")
        self.assertEqual(len(features), 5)
        self.assertTrue(all(0 <= value <= 1 for value in features))

    def test_given_no_dataset_when_requested_then_reproducible_starter_data_is_returned(self):
        self.assertEqual(make_dataset(4, 12), make_dataset(4, 12))
        self.assertEqual(len(make_dataset(4, 12)), 4)

    def test_rnn_and_gan_return_three_ink_controls(self):
        example = make_dataset(1)[0]
        self.assertEqual(len(InkRNN().predict(example.features)), 3)
        self.assertEqual(len(InkGAN().generate(example.features)), 3)

    def test_dqn_returns_a_known_transform_action(self):
        self.assertIn(InkDQN().choose(), InkDQN.ACTIONS)

    def test_dry_run_reports_all_three_training_tracks(self):
        result = dry_run(2, 5)
        self.assertEqual(result["dataset_size"], 5)
        self.assertEqual(len(result["history"]), 2)
        self.assertTrue(all(set(("rnn_loss", "gan_loss", "dqn_loss")) <= set(row) for row in result["history"]))


if __name__ == "__main__":
    unittest.main()