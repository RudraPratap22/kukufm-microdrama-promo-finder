import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from src.models import Block, Scene, Episode


def parse_timestamp(ts_str: str) -> float:
    """Convert mm:ss.xx or m:ss.xx timestamp string to float seconds."""
    ts_str = ts_str.strip()
    if ":" in ts_str:
        parts = ts_str.split(":")
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    return float(ts_str)


def format_timestamp(seconds: float) -> str:
    """Convert float seconds to mm:ss.xx format."""
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    rem_seconds = seconds % 60
    return f"{minutes:02d}:{rem_seconds:05.2f}"


def parse_block_tag(raw_tag: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Parse the block header inside **...**.
    Returns: (block_type, speaker, emotion_cue)
    """
    clean = raw_tag.strip().rstrip(":").strip()
    clean_upper = clean.upper()

    if clean_upper == "CAMERA":
        return "CAMERA", None, None
    elif clean_upper == "ACTION":
        return "ACTION", None, None
    elif clean_upper in ("SOUND FX", "SOUND_FX", "SOUNDFX", "SFX"):
        return "SOUND_FX", None, None

    # Character block, e.g. "LADY JULIANA (Confused, Breathless)" or "GERALD"
    if "(" in clean and clean.endswith(")"):
        paren_idx = clean.find("(")
        speaker = clean[:paren_idx].strip()
        emotion = clean[paren_idx + 1 : -1].strip()
        return "CHARACTER", speaker, emotion
    else:
        return "CHARACTER", clean, None


def parse_screenplay_text(raw_text: str, filename: str = "") -> Episode:
    """Parse the raw text of a single episode screenplay."""
    # 1. Header parsing
    ep_num_match = re.search(r"EPISODE\s+NUMBER:\s*Episode\s*(\d+)", raw_text, re.IGNORECASE)
    ep_num = int(ep_num_match.group(1)) if ep_num_match else 0

    logline_match = re.search(r"EPISODE\s+LOGLINE:\s*(.*?)(?=\n[A-Z\s]+:|\n---|\Z)", raw_text, re.DOTALL | re.IGNORECASE)
    logline = logline_match.group(1).strip() if logline_match else ""

    continuity_match = re.search(r"CONTINUITY\s+NOTES:\s*(.*?)(?=\n[A-Z\s]+:|\n---|\Z)", raw_text, re.DOTALL | re.IGNORECASE)
    continuity = continuity_match.group(1).strip() if continuity_match else ""

    # 2. Extract script content between delimiters
    script_match = re.search(
        r"---\s*SCRIPT\s+START\s*---\s*\n(.*?)\n\s*---\s*SCRIPT\s+END\s*---",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    script_body = script_match.group(1) if script_match else raw_text

    # 3. Split by scenes: "SCENE 01", "SCENE 1", etc.
    scene_splits = list(re.finditer(r"(?i)(?:^|\n)\s*SCENE\s+(\d+)\s*\n", script_body))

    scenes: List[Scene] = []
    block_pattern = re.compile(
        r"\*\*([^*]+)\*\*[:]?\s*\n\s*Start:\s*\(([0-9:.]+)\)\s*\|\s*End:\s*\(([0-9:.]+)\)\s*\n",
        re.IGNORECASE,
    )

    for i, match in enumerate(scene_splits):
        s_num = int(match.group(1))
        start_pos = match.end()
        end_pos = scene_splits[i + 1].start() if i + 1 < len(scene_splits) else len(script_body)
        scene_chunk = script_body[start_pos:end_pos]

        # Scene metadata (case-insensitive)
        heading_m = re.search(r"(?i)SCENE\s+HEADING:\s*([^\n]+)", scene_chunk)
        heading = heading_m.group(1).strip() if heading_m else "UNTITLED SCENE"

        s_start_m = re.search(r"(?i)SCENE\s+START:\s*([0-9:.]+)", scene_chunk)
        s_end_m = re.search(r"(?i)SCENE\s+END:\s*([0-9:.]+)", scene_chunk)

        s_start_str = s_start_m.group(1).strip() if s_start_m else "00:00.00"
        s_end_str = s_end_m.group(1).strip() if s_end_m else s_start_str
        s_start = parse_timestamp(s_start_str)
        s_end = parse_timestamp(s_end_str)

        # Actions and dialogue portion
        ad_marker = re.search(r"(?i)ACTIONS?\s+AND\s+DIALOGUE:\s*\n", scene_chunk)
        blocks_text = scene_chunk[ad_marker.end():] if ad_marker else scene_chunk

        # Parse blocks
        block_matches = list(block_pattern.finditer(blocks_text))
        scene_blocks: List[Block] = []

        for b_i, b_match in enumerate(block_matches):
            raw_tag = b_match.group(1)
            b_start_str = b_match.group(2)
            b_end_str = b_match.group(3)

            b_start = parse_timestamp(b_start_str)
            b_end = parse_timestamp(b_end_str)

            # Block text is from end of this match to start of next match or end of scene blocks
            text_start = b_match.end()
            text_end = block_matches[b_i + 1].start() if b_i + 1 < len(block_matches) else len(blocks_text)
            body_text = blocks_text[text_start:text_end].strip()

            b_type, speaker, emotion = parse_block_tag(raw_tag)

            # Dialogue detection: character block whose text is wrapped in quotes
            is_dlg = False
            if b_type == "CHARACTER":
                stripped = body_text.strip()
                if (stripped.startswith('"') and stripped.endswith('"')) or (
                    stripped.startswith('“') and stripped.endswith('”')
                ):
                    is_dlg = True

            scene_blocks.append(
                Block(
                    type=b_type,
                    speaker=speaker,
                    emotion=emotion,
                    start=b_start,
                    end=b_end,
                    start_str=b_start_str,
                    end_str=b_end_str,
                    text=body_text,
                    is_dialogue=is_dlg,
                    scene_num=s_num,
                )
            )

        scenes.append(
            Scene(
                scene_num=s_num,
                heading=heading,
                start=s_start,
                end=s_end,
                start_str=s_start_str,
                end_str=s_end_str,
                blocks=scene_blocks,
            )
        )

    return Episode(
        num=ep_num,
        filename=filename,
        logline=logline,
        continuity=continuity,
        scenes=scenes,
    )


def parse_episode_file(filepath: str | Path) -> Episode:
    """Read and parse an episode file from path."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return parse_screenplay_text(content, filename=path.name)


def get_lines(episode: Episode, start: float, end: float) -> List[Block]:
    """
    Retrieve verbatim blocks that overlap with the interval [start, end].
    Option B (Window Intersection):
    A block is included if block.end > start and block.start < end.
    """
    selected: List[Block] = []
    for b in episode.all_blocks:
        if b.end > start and b.start < end:
            selected.append(b)
    return selected


def format_episode_for_llm(episode: Episode, include_camera: bool = False) -> str:
    """Format an episode's scenes and blocks for the candidate extraction LLM prompt."""
    lines: List[str] = []
    for scene in episode.scenes:
        lines.append(f"--- SCENE {scene.scene_num:02d}: {scene.heading} ({scene.start_str} - {scene.end_str}) ---")
        for b in scene.blocks:
            line_str = b.to_formatted_line(include_camera=include_camera)
            if line_str:
                lines.append(line_str)
                lines.append("")  # Blank line separator
    return "\n".join(lines).strip()
