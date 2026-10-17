from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from training.config import CLASS_NAMES
from training.dataset import build_manifest, extract_patient_id, make_tf_dataset
from training.model import build_cnn
from utils.preprocessing import preprocess_image


class DatasetPipelineTests(unittest.TestCase):
    def _write_image(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (32, 32), color=127).save(path)

    def _create_fixture(self, root: Path) -> None:
        for split in ("train", "test", "val"):
            for class_name in CLASS_NAMES:
                (root / split / class_name).mkdir(parents=True, exist_ok=True)

        for patient in range(1, 6):
            for image_number in (1, 2):
                self._write_image(
                    root
                    / "train"
                    / "NORMAL"
                    / f"NORMAL2-IM-{patient:04d}-{image_number:04d}.jpeg"
                )
                self._write_image(
                    root
                    / "train"
                    / "PNEUMONIA"
                    / f"person{patient}_bacteria_{image_number}.jpeg"
                )

        self._write_image(root / "test" / "NORMAL" / "IM-9001-0001.jpeg")
        self._write_image(root / "test" / "PNEUMONIA" / "person9001_virus_1.jpeg")
        self._write_image(root / "val" / "NORMAL" / "IM-8001-0001.jpeg")
        self._write_image(root / "val" / "PNEUMONIA" / "person8001_virus_1.jpeg")

    def test_patient_identifiers(self) -> None:
        self.assertEqual(
            extract_patient_id(Path("NORMAL2-IM-0007-0002.jpeg")), "normal2-im-0007"
        )
        self.assertEqual(extract_patient_id(Path("person42_bacteria_3.jpeg")), "person42")

    def test_manifest_has_no_train_validation_patient_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "chest_xray"
            self._create_fixture(root)
            manifest, resolved = build_manifest(root)
            self.assertEqual(resolved, root.resolve())
            train_patients = set(manifest.loc[manifest["split"] == "train", "patient_id"])
            validation_patients = set(
                manifest.loc[manifest["split"] == "validation", "patient_id"]
            )
            self.assertFalse(train_patients.intersection(validation_patients))
            self.assertTrue(
                (manifest.loc[manifest["split"] == "test", "source_split"] == "test").all()
            )

    def test_dashboard_and_training_preprocessing_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "chest_xray"
            self._create_fixture(root)
            manifest, _ = build_manifest(root)
            dataset = make_tf_dataset(
                manifest, root, "validation", batch_size=2, shuffle=False
            )
            training_images, _ = next(iter(dataset))
            first_row = (
                manifest[manifest["split"] == "validation"]
                .sort_values("relative_path")
                .iloc[0]
            )
            image_path = root / Path(first_row["relative_path"])
            with Image.open(image_path) as source:
                image = source.copy()
            fake_model = type("FakeModel", (), {"input_shape": (None, 224, 224, 1)})()
            dashboard_batch = preprocess_image(
                image, fake_model, encoded_bytes=image_path.read_bytes()
            )
            self.assertEqual(training_images.shape, (2, 224, 224, 1))
            self.assertEqual(str(training_images.dtype), "<dtype: 'float32'>")
            np.testing.assert_allclose(
                training_images[0].numpy(), dashboard_batch[0], rtol=0, atol=0
            )


class ModelContractTests(unittest.TestCase):
    def test_notebook_architecture(self) -> None:
        model = build_cnn()
        self.assertEqual(model.input_shape, (None, 224, 224, 1))
        self.assertEqual(model.output_shape, (None, 1))
        self.assertEqual(model.count_params(), 11_168_513)


if __name__ == "__main__":
    unittest.main()
