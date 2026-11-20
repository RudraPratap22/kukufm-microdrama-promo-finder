from typing import List, Dict, Optional
from src.models import Episode


def build_show_synopsis(episodes: List[Episode]) -> str:
    """
    Build the overarching show synopsis by concatenating episode loglines in order.
    Plain code, no LLM call required.
    """
    sorted_eps = sorted(episodes, key=lambda e: e.num)
    synopsis_lines = []
    for ep in sorted_eps:
        synopsis_lines.append(f"Episode {ep.num}: {ep.logline}")
    return "\n".join(synopsis_lines)


def get_episode_context(
    ep: Episode,
    episodes_by_num: Dict[int, Episode],
    show_synopsis: str,
) -> Dict[str, str]:
    """
    Assemble the context variables needed for the candidate extraction prompt.
    """
    prev_ep_num = ep.num - 1
    prev_logline = (
        episodes_by_num[prev_ep_num].logline
        if prev_ep_num in episodes_by_num
        else "N/A (This is Episode 1, the series premiere)."
    )

    return {
        "show_synopsis": show_synopsis,
        "prev_episode_logline": prev_logline,
        "episode_logline": ep.logline,
        "continuity_notes": ep.continuity or "None provided.",
    }
