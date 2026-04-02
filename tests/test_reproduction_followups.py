import unittest

import train_spotrna


class TrainCheckpointStateTests(unittest.TestCase):
    def test_update_best_checkpoint_state_uses_new_best_epoch(self):
        best_val_f1, best_threshold, best_epoch, is_new_best = (
            train_spotrna.update_best_checkpoint_state(
                best_val_f1=0.61,
                best_threshold=0.30,
                best_epoch=2,
                val_metrics={"f1": 0.72, "threshold": 0.45},
                epoch_idx=3,
            )
        )

        self.assertEqual(best_val_f1, 0.72)
        self.assertEqual(best_threshold, 0.45)
        self.assertEqual(best_epoch, 3)
        self.assertTrue(is_new_best)

    def test_update_best_checkpoint_state_keeps_existing_best(self):
        best_val_f1, best_threshold, best_epoch, is_new_best = (
            train_spotrna.update_best_checkpoint_state(
                best_val_f1=0.72,
                best_threshold=0.45,
                best_epoch=3,
                val_metrics={"f1": 0.70, "threshold": 0.40},
                epoch_idx=4,
            )
        )

        self.assertEqual(best_val_f1, 0.72)
        self.assertEqual(best_threshold, 0.45)
        self.assertEqual(best_epoch, 3)
        self.assertFalse(is_new_best)


if __name__ == "__main__":
    unittest.main()
