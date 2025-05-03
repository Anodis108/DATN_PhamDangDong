from __future__ import annotations

import csv
import json
import sys
from functools import cached_property
from pathlib import Path
from typing import List
from typing import Optional

import mediapipe as mp
from common.bases import BaseModel
from common.bases import BaseService
from common.logs import get_logger
from common.settings import Settings

sys.path.append(str(Path(__file__).parent.parent))
logger = get_logger(__name__)


class CSVWriterInput(BaseModel):
    """Input data for CSV writing."""
    pose_landmarks_list: Optional[List[mp.tasks.python.vision.PoseLandmarkerResult]] = None
    distances: Optional[List[float]] = None
    pixel_per_cm: Optional[float] = None
    landmarks_csv_filename: Optional[str] = None
    distances_csv_filename: Optional[str] = None


class CSVWriterOutput(BaseModel):
    """Output status of CSV writing."""
    landmarks_csv_written: bool = False
    distances_csv_written: bool = False


class CSVWriterModel(BaseService):
    """Model to write pose landmarks and distances to CSV files."""
    settings: Settings

    @cached_property
    def body_parts(self) -> dict[int, str]:
        """Load body parts mapping from JSON file."""
        with open(self.settings.write_csv.body_parts_path) as f:
            return {int(k): v for k, v in json.load(f).items()}

    def _write_pose_landmarks(
        self,
        pose_landmarks_list: List[mp.tasks.python.vision.PoseLandmarkerResult],
        csv_filename: str,
    ) -> bool:
        """Write pose landmarks to a CSV file."""
        try:
            with open(csv_filename, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=[
                        'Pose', 'X', 'Y', 'Z',
                    ],
                )
                if csvfile.tell() == 0:
                    writer.writeheader()

                for idx, pose_landmarks in enumerate(pose_landmarks_list):
                    for i, landmark in enumerate(pose_landmarks):
                        writer.writerow({
                            'Pose': f'Pose {idx + 1} - {self.body_parts.get(i, "Unknown")}',
                            'X': landmark.x,
                            'Y': landmark.y,
                            'Z': landmark.z,
                        })
                    writer.writerow({})  # Empty row between poses
            return True
        except Exception as e:
            logger.error(
                f'Failed to write landmarks to {csv_filename}: {str(e)}',
            )
            return False

    def _write_distances(
        self,
        distances: List[float],
        pixel_per_cm: float,
        csv_filename: str,
    ) -> bool:
        """Write distances and height to a CSV file."""
        try:
            with open(csv_filename, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if csvfile.tell() == 0:
                    writer.writerow(
                        ['Pose'] +
                        [f'Distance{i + 1} (cm)' for i in range(len(distances))] +
                        ['Height (cm)'],
                    )
                distances_cm = [
                    distance /
                    pixel_per_cm for distance in distances
                ]
                writer.writerow(['Pose 1'] + distances_cm +
                                [sum(distances_cm)])
            return True
        except Exception as e:
            logger.error(
                f'Failed to write distances to {csv_filename}: {str(e)}',
            )
            return False

    def process(self, inputs: CSVWriterInput) -> CSVWriterOutput:
        """Process input data and write to CSV files."""
        try:
            landmarks_written = False
            distances_written = False

            if inputs.pose_landmarks_list and inputs.landmarks_csv_filename:
                landmarks_written = self._write_pose_landmarks(
                    inputs.pose_landmarks_list,
                    inputs.landmarks_csv_filename,
                )

            if inputs.distances and inputs.pixel_per_cm and inputs.distances_csv_filename:
                distances_written = self._write_distances(
                    inputs.distances,
                    inputs.pixel_per_cm,
                    inputs.distances_csv_filename,
                )

            return CSVWriterOutput(
                landmarks_csv_written=landmarks_written,
                distances_csv_written=distances_written,
            )
        except Exception as e:
            logger.error(f'CSV writing failed: {str(e)}')
            raise
