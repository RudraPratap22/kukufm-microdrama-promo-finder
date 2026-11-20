import json
from pathlib import Path
from typing import Dict, List
from src.models import Episode
from src.parser import get_lines
from src.validator import ValidatedPromo


def write_promos_json(promos: List[ValidatedPromo], output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [p.to_dict() for p in promos]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"total_promos": len(promos), "promos": data}, f, indent=2)


def write_promos_md(
    promos: List[ValidatedPromo],
    episodes_by_num: Dict[int, Episode],
    output_path: str,
    ranking_strategy: str = "",
):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = [
        "# Top Microdrama Promos",
        "",
        "> Generated from screenplay analysis. Screenplay segments below are extracted **verbatim** from the source scripts via `get_lines()`.",
        "",
    ]

    if ranking_strategy:
        lines.extend([
            f"**Marketing & Ranking Strategy**: {ranking_strategy}",
            "",
        ])

    lines.extend([
        "---",
        "",
    ])

    for p in promos:
        ep = episodes_by_num.get(p.episode_num)
        segments_str = ", ".join(f"`{s.start_str} - {s.end_str}` ({s.duration:.1f}s)" for s in p.segments)

        lines.extend([
            f"## Rank #{p.rank}: Episode {p.episode_num:02d} (Candidate {p.candidate_id})",
            "",
            f"- **Ranking Reason**: {p.ranking_reason or p.why}",
            f"- **Total Duration**: {p.duration:.1f} seconds",
            f"- **Segments**: {segments_str}",
            f"- **Composite Score**: {p.composite_score:.2f} / 10.0 (Hook: {p.scores.get('hook_strength')}, Cliffhanger: {p.scores.get('cliffhanger_strength')}, Standalone: {p.scores.get('standalone_clarity')})",
            f"- **The 3-Second Hook**: {p.hook}",
            f"- **The Cliffhanger**: {p.cliffhanger}",
            f"- **Why It Works**: {p.why}",
            f"- **New Viewer Risk**: {p.confusion_risk or 'None noted.'}",
            "",
            "### Quality Check",
            "Hook [ ]  Cliffhanger [ ]  Standalone [ ]",
            "",
            "### Verbatim Screenplay Lines",
            "",
        ])

        if ep:
            for s_idx, seg in enumerate(p.segments, start=1):
                if len(p.segments) > 1:
                    lines.append(f"#### Segment {s_idx} (`{seg.start_str} - {seg.end_str}`)")
                    lines.append("")

                blocks = get_lines(ep, seg.start, seg.end)
                if not blocks:
                    lines.append("*(No blocks found in range)*\n")
                else:
                    lines.append("```text")
                    for b in blocks:
                        cue = f" ({b.emotion})" if b.emotion else ""
                        speaker_label = f"{b.speaker}{cue}" if b.speaker else b.type
                        lines.append(f"[{b.start_str} - {b.end_str}] {speaker_label}:")
                        lines.append(f"  {b.text}")
                        lines.append("")
                    lines.append("```")
                    lines.append("")
        else:
            lines.append("*(Episode data not loaded)*\n")

        lines.append("---\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
