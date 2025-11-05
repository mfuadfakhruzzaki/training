"""Training script for sensor dataset models.

This script trains three simple models based on the available dataset:
1. Grade prediction using a nearest-centroid classifier in standardized space.
2. An anomaly detector using z-score thresholds derived from normal observations.
3. A pump control model combining rule-based activation with linear regression for duration.

The models are saved as pickle files under the ``models`` directory.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

DATA_FILE = "sensor_dataset_labeled_3grade.csv"
MODEL_DIR = "models"
DEFAULT_BUNDLE_NAME = "models_bundle.zip"


def read_dataset(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]


@dataclass
class FeatureScaler:
    means: Tuple[float, ...]
    stds: Tuple[float, ...]

    @classmethod
    def fit(cls, vectors: Sequence[Sequence[float]]) -> "FeatureScaler":
        transposed = list(zip(*vectors))
        means = []
        stds = []
        for column in transposed:
            column_values = list(column)
            column_mean = sum(column_values) / len(column_values)
            variance = sum((value - column_mean) ** 2 for value in column_values) / len(column_values)
            column_std = math.sqrt(variance) if variance > 0 else 0.0
            means.append(column_mean)
            stds.append(column_std)
        return cls(tuple(means), tuple(stds))

    def transform(self, vector: Sequence[float]) -> Tuple[float, ...]:
        scaled = []
        for value, mu, sigma in zip(vector, self.means, self.stds):
            if sigma == 0:
                scaled.append(0.0)
            else:
                scaled.append((value - mu) / sigma)
        return tuple(scaled)

    def transform_many(self, vectors: Iterable[Sequence[float]]) -> List[Tuple[float, ...]]:
        return [self.transform(vector) for vector in vectors]


@dataclass
class GradeCentroidModel:
    scaler: FeatureScaler
    centroids: Dict[str, Tuple[float, ...]]

    @classmethod
    def train(cls, rows: Sequence[Dict[str, str]]) -> "GradeCentroidModel":
        features = []
        labels = []
        for row in rows:
            feature_vector = (
                float(row["suhu"]),
                float(row["kelembaban"]),
                float(row["amonia"]),
            )
            features.append(feature_vector)
            labels.append(row["grade"])
        scaler = FeatureScaler.fit(features)
        scaled_features = scaler.transform_many(features)
        centroids: Dict[str, List[float]] = {}
        counts: Dict[str, int] = {}
        for vector, label in zip(scaled_features, labels):
            if label not in centroids:
                centroids[label] = [0.0 for _ in vector]
                counts[label] = 0
            centroids[label] = [acc + value for acc, value in zip(centroids[label], vector)]
            counts[label] += 1
        centroid_vectors = {
            label: tuple(value / counts[label] for value in sums)
            for label, sums in centroids.items()
        }
        return cls(scaler=scaler, centroids=centroid_vectors)

    def predict(self, feature_vector: Sequence[float]) -> str:
        scaled_vector = self.scaler.transform(feature_vector)
        best_label = None
        best_distance = None
        for label, centroid in self.centroids.items():
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(scaled_vector, centroid)))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_label = label
        assert best_label is not None
        return best_label


@dataclass
class AnomalyZScoreModel:
    feature_means: Tuple[float, ...]
    feature_stds: Tuple[float, ...]
    warning_factor: float = 2.0
    anomaly_factor: float = 3.0

    @classmethod
    def train(cls, rows: Sequence[Dict[str, str]]) -> "AnomalyZScoreModel":
        normal_rows = [row for row in rows if row["anomaly_status"].lower() == "normal"]
        # Fall back to entire dataset if normal samples are scarce
        source = normal_rows if normal_rows else rows
        features = [
            (
                float(row["suhu"]),
                float(row["kelembaban"]),
                float(row["amonia"]),
            )
            for row in source
        ]
        scaler = FeatureScaler.fit(features)
        return cls(feature_means=scaler.means, feature_stds=scaler.stds)

    def score(self, feature_vector: Sequence[float]) -> Tuple[str, Tuple[float, ...]]:
        z_scores = []
        for value, mu, sigma in zip(feature_vector, self.feature_means, self.feature_stds):
            if sigma == 0:
                z_scores.append(0.0)
            else:
                z_scores.append((value - mu) / sigma)
        max_abs = max(abs(z) for z in z_scores)
        if max_abs >= self.anomaly_factor:
            status = "anomaly"
        elif max_abs >= self.warning_factor:
            status = "warning"
        else:
            status = "normal"
        return status, tuple(z_scores)


@dataclass
class PumpControlModel:
    temperature_threshold: float
    humidity_threshold: float
    duration_coefficients: Tuple[float, float, float]
    min_duration: float
    max_duration: float

    @classmethod
    def train(cls, rows: Sequence[Dict[str, str]]) -> "PumpControlModel":
        temp_on: List[float] = []
        temp_off: List[float] = []
        hum_on: List[float] = []
        hum_off: List[float] = []
        regression_features: List[Tuple[float, float]] = []
        regression_targets: List[float] = []
        min_duration = math.inf
        max_duration = -math.inf

        for row in rows:
            temp = float(row["suhu"])
            humid = float(row["kelembaban"])
            duration = float(row["pump_duration_s"])
            pump_on = row["pump_on"] == "True"
            min_duration = min(min_duration, duration)
            max_duration = max(max_duration, duration)
            if pump_on:
                temp_on.append(temp)
                hum_on.append(humid)
                regression_features.append((temp, humid))
                regression_targets.append(duration)
            else:
                temp_off.append(temp)
                hum_off.append(humid)

        def average(values: List[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        temperature_threshold = (average(temp_on) + average(temp_off)) / 2.0 if temp_off else average(temp_on)
        humidity_threshold = (average(hum_on) + average(hum_off)) / 2.0 if hum_off else average(hum_on)

        coefficients = cls._fit_linear_regression(regression_features, regression_targets)
        return cls(
            temperature_threshold=temperature_threshold,
            humidity_threshold=humidity_threshold,
            duration_coefficients=coefficients,
            min_duration=min_duration,
            max_duration=max_duration,
        )

    @staticmethod
    def _fit_linear_regression(features: Sequence[Tuple[float, float]], targets: Sequence[float]) -> Tuple[float, float, float]:
        if not features:
            return (0.0, 0.0, 0.0)
        # Normal equation components
        xtx = [[0.0 for _ in range(3)] for _ in range(3)]
        xty = [0.0 for _ in range(3)]
        for (temp, humid), target in zip(features, targets):
            vector = (1.0, temp, humid)
            for i in range(3):
                for j in range(3):
                    xtx[i][j] += vector[i] * vector[j]
                xty[i] += vector[i] * target
        return PumpControlModel._solve_3x3(xtx, xty)

    @staticmethod
    def _solve_3x3(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Tuple[float, float, float]:
        def det3(m: Sequence[Sequence[float]]) -> float:
            return (
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            )

        determinant = det3(matrix)
        if abs(determinant) < 1e-8:
            # Singular matrix; fall back to zeros
            return (0.0, 0.0, 0.0)

        coefficients: List[float] = []
        for column in range(3):
            substituted = [list(row) for row in matrix]
            for row_index in range(3):
                substituted[row_index][column] = vector[row_index]
            coefficients.append(det3(substituted) / determinant)
        return tuple(coefficients)  # type: ignore[return-value]

    def predict(self, temp: float, humid: float) -> Tuple[bool, float]:
        activate = temp >= self.temperature_threshold or humid <= self.humidity_threshold
        intercept, temp_coef, humid_coef = self.duration_coefficients
        predicted = intercept + temp_coef * temp + humid_coef * humid
        clamped = max(self.min_duration, min(self.max_duration, predicted))
        return activate, clamped


def ensure_model_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_pickle(path: Path, obj: object) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def bundle_artifacts(directory: Path, artifact_names: Sequence[str], bundle_path: Path) -> None:
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in artifact_names:
            bundle.write(directory / name, arcname=name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export sensor dataset models")
    parser.add_argument(
        "--data",
        default=DATA_FILE,
        help="Path to the labeled sensor dataset CSV (default: sensor_dataset_labeled_3grade.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=MODEL_DIR,
        help="Directory where trained model artifacts will be stored (default: models)",
    )
    parser.add_argument(
        "--no-bundle",
        action="store_true",
        help="Skip creation of a ZIP bundle containing all model artifacts",
    )
    parser.add_argument(
        "--bundle-name",
        default=DEFAULT_BUNDLE_NAME,
        help="Filename for the bundled ZIP archive (default: models_bundle.zip)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.data)
    output_dir = Path(args.output_dir)
    ensure_model_dir(output_dir)

    rows = read_dataset(str(dataset_path))

    grade_model = GradeCentroidModel.train(rows)
    anomaly_model = AnomalyZScoreModel.train(rows)
    pump_model = PumpControlModel.train(rows)

    artifacts = {
        "grade_classifier.pkl": grade_model,
        "anomaly_detector.pkl": anomaly_model,
        "pump_controller.pkl": pump_model,
    }

    for filename, model in artifacts.items():
        save_pickle(output_dir / filename, model)

    print("Models trained and saved to:")
    for filename in artifacts:
        print(f" - {output_dir / filename}")

    if not args.no_bundle:
        bundle_path = output_dir / args.bundle_name
        bundle_artifacts(output_dir, artifacts.keys(), bundle_path)
        print(f"Bundled archive created at: {bundle_path}")


if __name__ == "__main__":
    main()