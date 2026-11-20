from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Block:
    type: str  # "CAMERA", "ACTION", "SOUND_FX", "CHARACTER"
    speaker: Optional[str]
    emotion: Optional[str]
    start: float  # Seconds
    end: float  # Seconds
    start_str: str  # Original mm:ss.xx string
    end_str: str  # Original mm:ss.xx string
    text: str
    is_dialogue: bool
    scene_num: int = 1

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_formatted_line(self, include_camera: bool = True) -> Optional[str]:
        if not include_camera and self.type == "CAMERA":
            return None
        if self.type == "CAMERA":
            return f"**CAMERA:** ({self.start_str} - {self.end_str})\n{self.text}"
        elif self.type == "ACTION":
            return f"**ACTION:** ({self.start_str} - {self.end_str})\n{self.text}"
        elif self.type == "SOUND_FX":
            return f"**SOUND FX:** ({self.start_str} - {self.end_str})\n{self.text}"
        else:
            cue = f" ({self.emotion})" if self.emotion else ""
            return f"**{self.speaker}{cue}:** ({self.start_str} - {self.end_str})\n{self.text}"


@dataclass
class Scene:
    scene_num: int
    heading: str
    start: float
    end: float
    start_str: str
    end_str: str
    blocks: List[Block] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Episode:
    num: int
    filename: str
    logline: str
    continuity: str
    scenes: List[Scene] = field(default_factory=list)

    @property
    def duration(self) -> float:
        if not self.scenes:
            return 0.0
        return max(scene.end for scene in self.scenes)

    @property
    def all_blocks(self) -> List[Block]:
        blocks: List[Block] = []
        for s in self.scenes:
            blocks.extend(s.blocks)
        return blocks
