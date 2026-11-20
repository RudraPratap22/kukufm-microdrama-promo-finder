import unittest
from pathlib import Path

from run import candidate_raw_path
from src.models import Block, Episode, Scene
from src.validator import CandidateValidator


class CacheAndValidatorTests(unittest.TestCase):
    def setUp(self):
        self.episode = Episode(
            num=1,
            filename="episode-001.txt",
            logline="",
            continuity="",
            scenes=[
                Scene(
                    scene_num=1,
                    heading="INT. ROOM",
                    start=0.0,
                    end=40.0,
                    start_str="00:00.00",
                    end_str="00:40.00",
                    blocks=[
                        Block("CHARACTER", "A", None, 5.0, 8.0, "00:05.00", "00:08.00", '"Hello."', True),
                        Block("ACTION", None, None, 10.0, 30.0, "00:10.00", "00:30.00", "Moves across the room.", False),
                        Block("CHARACTER", "B", None, 31.0, 40.0, "00:31.00", "00:40.00", '"Stop!"', True),
                    ],
                )
            ],
        )

    def test_candidate_raw_path_varies_by_model_and_camera_mode(self):
        raw_dir = Path("output/raw")
        qwen_nocam = candidate_raw_path(raw_dir, 1, "qwen/qwen3.8-27b", False)
        qwen_cam = candidate_raw_path(raw_dir, 1, "qwen/qwen3.8-27b", True)
        other_model = candidate_raw_path(raw_dir, 1, "provider/model-x", False)
        sanitized_collision = candidate_raw_path(raw_dir, 1, "qwen-qwen3.8-27b", False)

        self.assertNotEqual(qwen_nocam, qwen_cam)
        self.assertNotEqual(qwen_nocam, other_model)
        self.assertNotEqual(qwen_nocam, sanitized_collision)
        self.assertIn("find_candidates_v1", qwen_nocam.name)

    def test_valid_timestamps_snap_to_required_anchors(self):
        validator = CandidateValidator()
        parsed = {
            "candidates": [
                {
                    "candidate_id": 1,
                    "segments": [{"start": "00:10.00", "end": "00:31.00"}],
                    "scores": {"hook_strength": 8, "cliffhanger_strength": 8, "standalone_clarity": 8},
                }
            ]
        }

        promos = validator.validate_episode_candidates(self.episode, parsed)

        self.assertEqual(len(promos), 1)
        self.assertEqual(promos[0].segments[0].start_str, "00:05.00")
        self.assertEqual(promos[0].segments[0].end_str, "00:30.00")

    def test_timestamp_not_in_screenplay_is_rejected(self):
        validator = CandidateValidator()
        parsed = {
            "candidates": [
                {
                    "candidate_id": 1,
                    "segments": [{"start": "00:12.00", "end": "00:30.00"}],
                    "scores": {"hook_strength": 8, "cliffhanger_strength": 8, "standalone_clarity": 8},
                }
            ]
        }

        promos = validator.validate_episode_candidates(self.episode, parsed)

        self.assertEqual(promos, [])
        self.assertIn("does not appear", validator.rejection_log[0]["reason"])

    def test_end_timestamp_not_in_screenplay_is_rejected(self):
        validator = CandidateValidator()
        parsed = {
            "candidates": [
                {
                    "candidate_id": 1,
                    "segments": [{"start": "00:05.00", "end": "00:12.00"}],
                    "scores": {"hook_strength": 8, "cliffhanger_strength": 8, "standalone_clarity": 8},
                }
            ]
        }

        promos = validator.validate_episode_candidates(self.episode, parsed)

        self.assertEqual(promos, [])
        self.assertIn("does not appear", validator.rejection_log[0]["reason"])


if __name__ == "__main__":
    unittest.main()