import argparse
import os
import tempfile
import unittest
from unittest import mock

import torch

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

    def test_main_writes_updated_best_metadata_to_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            datasets_dir = os.path.join(temp_dir, "datasets")
            os.makedirs(datasets_dir)
            open(os.path.join(datasets_dir, "bpRNA_dataset.zip"), "wb").close()

            output_dir = os.path.join(temp_dir, "outputs")
            resume_checkpoint = os.path.join(temp_dir, "resume.pt")
            run_name = "resume-regression"
            args = argparse.Namespace(
                phase="pretrain",
                datasets_dir=datasets_dir,
                preset="paper-small",
                epochs=3,
                batch_size=1,
                learning_rate=1e-3,
                positive_weight="auto",
                threshold=0.335,
                device="cpu",
                amp=False,
                amp_dtype="bf16",
                seed=7,
                init_checkpoint="",
                output_dir=output_dir,
                run_name=run_name,
                max_train_samples=0,
                max_val_samples=0,
                max_test_samples=0,
                save_every_epoch=False,
                resume_checkpoint=resume_checkpoint,
                num_workers=0,
                drop_multiplets=False,
                log_interval=0,
                threshold_min=0.1,
                threshold_max=0.2,
                threshold_step=0.05,
                standardize_input=False,
            )
            val_metrics = {
                "loss": 0.2,
                "f1": 0.72,
                "mcc": 0.5,
                "threshold": 0.45,
            }
            test_metrics = {
                "loss": 0.1,
                "f1": 0.7,
                "precision": 0.71,
                "sensitivity": 0.69,
                "mcc": 0.4,
            }

            resumed_model = torch.nn.Linear(1, 1)
            resumed_optimizer = torch.optim.Adam(
                resumed_model.parameters(), lr=args.learning_rate
            )
            torch.save(
                {
                    "state_dict": resumed_model.state_dict(),
                    "optimizer_state_dict": resumed_optimizer.state_dict(),
                    "args": {},
                    "model_config": {},
                    "history": [{"epoch": 2, "val": {"f1": 0.61}}],
                    "epoch": 2,
                    "best_val_f1": 0.61,
                    "best_threshold": 0.30,
                    "best_epoch": 2,
                    "feature_mean": None,
                    "feature_std": None,
                },
                resume_checkpoint,
            )

            with (
                mock.patch.object(train_spotrna, "parse_args", return_value=args),
                mock.patch.object(train_spotrna, "set_seed"),
                mock.patch.object(
                    train_spotrna,
                    "PaperInspiredSPOTRNA",
                    side_effect=lambda **_: torch.nn.Linear(1, 1),
                ),
                mock.patch.object(
                    train_spotrna,
                    "RNAPairDataset",
                    side_effect=[object(), object(), object()],
                ),
                mock.patch.object(
                    train_spotrna,
                    "build_dataloader",
                    side_effect=[object(), object(), object()],
                ),
                mock.patch.object(train_spotrna, "train_one_epoch", return_value=1.23),
                mock.patch.object(
                    train_spotrna,
                    "search_best_threshold",
                    return_value=val_metrics,
                ),
                mock.patch.object(
                    train_spotrna,
                    "evaluate_model",
                    return_value=test_metrics,
                ),
            ):
                train_spotrna.main()

            last_checkpoint = torch.load(
                os.path.join(output_dir, run_name, "last.pt"),
                map_location="cpu",
                weights_only=False,
            )

            self.assertEqual(last_checkpoint["best_val_f1"], 0.72)
            self.assertEqual(last_checkpoint["best_threshold"], 0.45)
            self.assertEqual(last_checkpoint["best_epoch"], 3)


if __name__ == "__main__":
    unittest.main()
