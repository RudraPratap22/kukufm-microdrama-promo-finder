import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from src.models import Block, Episode
from src.parser import format_timestamp, parse_timestamp


@dataclass
class SegmentCandidate:
    start: float
    end: float
    start_str: str
    end_str: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class ValidatedPromo:
    episode_num: int
    candidate_id: int
    segments: List[SegmentCandidate]
    duration: float
    hook: str
    cliffhanger: str
    why: str
    confusion_risk: str
    scores: Dict[str, float]
    composite_score: float
    rank: int = 0
    ranking_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "episode_num": self.episode_num,
            "candidate_id": self.candidate_id,
            "ranking_reason": self.ranking_reason,
            "segments": [
                {
                    "start": s.start_str,
                    "end": s.end_str,
                    "duration_seconds": round(s.duration, 2),
                }
                for s in self.segments
            ],
            "total_duration_seconds": round(self.duration, 2),
            "hook": self.hook,
            "cliffhanger": self.cliffhanger,
            "why_it_works": self.why,
            "new_viewer_confusion_risk": self.confusion_risk,
            "scores": self.scores,
            "composite_score": round(self.composite_score, 2),
        }


class CandidateValidator:
    def __init__(self):
        self.rejection_log: List[Dict[str, Any]] = []

    def log_rejection(self, episode_num: int, candidate_id: Any, reason: str, details: Any = None):
        self.rejection_log.append({
            "episode_num": episode_num,
            "candidate_id": candidate_id,
            "reason": reason,
            "details": details,
        })

    def snap_start_time(self, raw_start: float, episode: Episode) -> Optional[Tuple[float, str]]:
        """
        Snap start time to nearest dialogue line start or scene start (never camera blocks).
        """
        candidates: List[Tuple[float, str]] = []

        # Scene starts
        for scene in episode.scenes:
            candidates.append((scene.start, scene.start_str))

        # Dialogue line starts only
        for block in episode.all_blocks:
            if block.is_dialogue:
                candidates.append((block.start, block.start_str))

        if not candidates:
            return None

        best = min(candidates, key=lambda c: abs(c[0] - raw_start))
        return best[0], best[1]

    def snap_end_time(self, raw_end: float, episode: Episode) -> Optional[Tuple[float, str]]:
        """
        Snap end time to nearest block end.
        """
        candidates: List[Tuple[float, str]] = []
        for block in episode.all_blocks:
            candidates.append((block.end, block.end_str))

        if not candidates:
            return None

        best = min(candidates, key=lambda c: abs(c[0] - raw_end))
        return best[0], best[1]

    def check_overlap(self, segs_a: List[SegmentCandidate], segs_b: List[SegmentCandidate]) -> bool:
        """Return True if any segment in segs_a overlaps with any segment in segs_b."""
        for a in segs_a:
            for b in segs_b:
                if max(a.start, b.start) < min(a.end, b.end):
                    return True
        return False

    def validate_episode_candidates(
        self,
        episode: Episode,
        raw_candidates_json: Dict[str, Any],
    ) -> List[ValidatedPromo]:
        """
        Validate and snap candidates for a single episode.
        Filters invalid candidates and prevents intra-episode overlaps.
        """
        if not raw_candidates_json.get("has_viable_promo", True):
            self.log_rejection(
                episode.num,
                "ALL",
                "Model declared no viable promo for this episode",
                raw_candidates_json.get("evaluation_notes", ""),
            )
            return []

        candidates_data = raw_candidates_json.get("candidates", [])
        if not candidates_data:
            self.log_rejection(episode.num, "ALL", "No candidates in response")
            return []

        episode_timestamps = {
            round(timestamp, 2)
            for scene in episode.scenes
            for timestamp in (scene.start, scene.end)
        }
        episode_timestamps.update(
            round(timestamp, 2)
            for block in episode.all_blocks
            for timestamp in (block.start, block.end)
        )

        scored_candidates: List[Tuple[float, ValidatedPromo]] = []

        for c_idx, c_data in enumerate(candidates_data):
            c_id = c_data.get("candidate_id", c_idx + 1)
            raw_segments = c_data.get("segments", [])

            if not raw_segments:
                self.log_rejection(episode.num, c_id, "Candidate has empty segments list")
                continue

            snapped_segments: List[SegmentCandidate] = []
            segment_error = None

            for s_idx, s_item in enumerate(raw_segments):
                s_start_str = s_item.get("start", "")
                s_end_str = s_item.get("end", "")

                try:
                    s_start = parse_timestamp(s_start_str)
                    s_end = parse_timestamp(s_end_str)
                except Exception as e:
                    segment_error = f"Malformed timestamp in segment {s_idx}: {e}"
                    break

                # Boundary check against episode duration
                if s_start < 0.0 or s_end > (episode.duration + 1.0):
                    segment_error = f"Timestamps [{s_start_str} - {s_end_str}] out of episode bounds (0 to {format_timestamp(episode.duration)})"
                    break

                if s_end <= s_start:
                    segment_error = f"Segment {s_idx} end <= start ({s_start_str} to {s_end_str})"
                    break

                if round(s_start, 2) not in episode_timestamps:
                    segment_error = f"Start timestamp {s_start_str} does not appear in the episode screenplay"
                    break

                if round(s_end, 2) not in episode_timestamps:
                    segment_error = f"End timestamp {s_end_str} does not appear in the episode screenplay"
                    break

                # Snap timestamps
                snapped_start = self.snap_start_time(s_start, episode)
                snapped_end = self.snap_end_time(s_end, episode)

                if snapped_start is None:
                    segment_error = "No dialogue or scene-start anchor is available for the segment start"
                    break

                if snapped_end is None:
                    segment_error = "No block-end anchor is available for the segment end"
                    break

                snapped_s, snapped_s_str = snapped_start
                snapped_e, snapped_e_str = snapped_end

                if snapped_e <= snapped_s:
                    segment_error = f"Segment {s_idx} end <= start after snapping ({snapped_s_str} to {snapped_e_str})"
                    break

                snapped_segments.append(
                    SegmentCandidate(
                        start=snapped_s,
                        end=snapped_e,
                        start_str=snapped_s_str,
                        end_str=snapped_e_str,
                    )
                )

            if segment_error:
                self.log_rejection(episode.num, c_id, segment_error)
                continue

            # Verify segment ordering
            is_ordered = True
            for i in range(len(snapped_segments) - 1):
                if snapped_segments[i].end > snapped_segments[i + 1].start:
                    is_ordered = False
                    break

            if not is_ordered:
                self.log_rejection(episode.num, c_id, "Segments are out of chronological order or overlap each other")
                continue

            # Verify total duration (20s to 60s)
            total_dur = sum(s.duration for s in snapped_segments)
            if total_dur < 20.0 or total_dur > 60.0:
                self.log_rejection(
                    episode.num,
                    c_id,
                    f"Total duration {total_dur:.1f}s outside required 20-60s window",
                    {"segments": [f"{s.start_str}-{s.end_str}" for s in snapped_segments]},
                )
                continue

            # Compute composite score
            scores = c_data.get("scores", {})
            hook_s = float(scores.get("hook_strength", 5))
            cliff_s = float(scores.get("cliffhanger_strength", 5))
            clarity_s = float(scores.get("standalone_clarity", 5))
            composite = (hook_s * 0.4) + (cliff_s * 0.4) + (clarity_s * 0.2)

            promo = ValidatedPromo(
                episode_num=episode.num,
                candidate_id=c_id,
                segments=snapped_segments,
                duration=total_dur,
                hook=c_data.get("hook_description", ""),
                cliffhanger=c_data.get("cliffhanger_description", ""),
                why=c_data.get("why_it_works", ""),
                confusion_risk=c_data.get("new_viewer_confusion_risk", ""),
                scores=scores,
                composite_score=composite,
            )
            scored_candidates.append((composite, promo))

        # Sort by composite score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        accepted_promos: List[ValidatedPromo] = []
        for score, candidate in scored_candidates:
            # Check overlap with already accepted promos from this episode
            has_overlap = False
            for accepted in accepted_promos:
                if self.check_overlap(candidate.segments, accepted.segments):
                    has_overlap = True
                    self.log_rejection(
                        episode.num,
                        candidate.candidate_id,
                        f"Overlaps with higher-scoring candidate #{accepted.candidate_id} in episode {episode.num}",
                    )
                    break

            if not has_overlap:
                accepted_promos.append(candidate)

        return accepted_promos

    def write_validation_log(self, output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_rejections": len(self.rejection_log),
                    "rejections": self.rejection_log,
                },
                f,
                indent=2,
            )
