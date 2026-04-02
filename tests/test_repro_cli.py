import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_help(relative_path: str) -> str:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / relative_path), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


class ReproCliSurfaceTest(unittest.TestCase):
    def test_train_wrapper_exposes_real_training_options(self) -> None:
        output = run_help("repro/train.py")
        self.assertIn("--phase", output)
        self.assertIn("--datasets-dir", output)
        self.assertIn("--init-checkpoint", output)

    def test_eval_wrapper_exposes_ensemble_options(self) -> None:
        output = run_help("repro/eval.py")
        self.assertIn("--checkpoints", output)
        self.assertIn("--output-json", output)
        self.assertIn("--drop-multiplets", output)

    def test_infer_wrapper_exposes_checkpoint_inference_options(self) -> None:
        output = run_help("repro/infer.py")
        self.assertIn("--checkpoint", output)
        self.assertIn("--input", output)
        self.assertIn("--output", output)


class OfficialDockerWrapperTest(unittest.TestCase):
    def test_absolute_repo_paths_are_rewritten_to_workspace_mount(self) -> None:
        input_path = REPO_ROOT / "sample_inputs" / "single_seq.fasta"
        output_path = REPO_ROOT / "outputs"
        script_path = REPO_ROOT / "official" / "docker" / "run_inference.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            capture_path = temp_path / "docker-args.txt"
            docker_path = temp_path / "docker"
            docker_path.write_text(
                '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$TEST_CAPTURE"\n',
                encoding="utf-8",
            )
            docker_path.chmod(0o755)

            env = os.environ.copy()
            env["TEST_CAPTURE"] = str(capture_path)
            env["PATH"] = f"{temp_dir}:{env['PATH']}"

            subprocess.run(
                [
                    "sh",
                    str(script_path),
                    str(input_path),
                    str(output_path),
                ],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            captured = capture_path.read_text(encoding="utf-8")
            self.assertIn("--inputs", captured)
            self.assertIn("--outputs", captured)
            self.assertIn("/workspace/sample_inputs/single_seq.fasta", captured)
            self.assertIn("/workspace/outputs", captured)


if __name__ == "__main__":
    unittest.main()
