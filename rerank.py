"""
Standalone re-ranker: loads all validated promos from output/promos.json,
calls the global LLM ranker, and rewrites promos.json + promos.md.
Run: python rerank.py --input "path/to/show folder" --out output
"""
import argparse
import json
from pathlib import Path

from src.llm import LLMClient
from src.validator import ValidatedPromo
from src.ranker import rank_promos_globally
from src.output_writer import write_promos_json, write_promos_md
from src.parser import parse_episode_file
from src.context import build_show_synopsis


def load_validated_promos(promos_json_path: Path):
    """
    Reconstruct ValidatedPromo objects from the saved promos.json.
    Maps JSON keys (why_it_works, new_viewer_confusion_risk, total_duration_seconds)
    back to dataclass fields (why, confusion_risk, duration).
    Segments are stored with start_str/end_str; we rebuild SegmentCandidate objects.
    """
    from src.validator import SegmentCandidate
    from src.parser import parse_timestamp, format_timestamp

    with open(promos_json_path, encoding="utf-8") as f:
        data = json.load(f)
    promos = []
    for d in data["promos"]:
        segs = []
        for s in d["segments"]:
            start_str = s["start"]
            end_str = s["end"]
            segs.append(SegmentCandidate(
                start=parse_timestamp(start_str),
                end=parse_timestamp(end_str),
                start_str=start_str,
                end_str=end_str,
            ))
        p = ValidatedPromo(
            episode_num=d["episode_num"],
            candidate_id=d["candidate_id"],
            segments=segs,
            duration=d["total_duration_seconds"],
            hook=d["hook"],
            cliffhanger=d["cliffhanger"],
            why=d["why_it_works"],
            confusion_risk=d.get("new_viewer_confusion_risk", ""),
            scores=d["scores"],
            composite_score=d["composite_score"],
            rank=d.get("rank", 0),
            ranking_reason=d.get("ranking_reason", ""),
        )
        promos.append(p)
    return promos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Show folder path")
    parser.add_argument("--out", default="output", help="Output folder")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    out_dir = Path(args.out)
    promos_json_path = out_dir / "promos.json"

    # Load config and create LLM client (reads config.yaml and .env automatically)
    llm = LLMClient(config_path=args.config)

    # Load promos from existing JSON
    print(f"[rerank] Loading promos from {promos_json_path}...")
    promos = load_validated_promos(promos_json_path)
    print(f"[rerank] Loaded {len(promos)} promos.")

    # Build synopsis from input folder
    show_dir = Path(args.input)
    eps = [parse_episode_file(f) for f in sorted(show_dir.glob("*.txt"))]
    synopsis = build_show_synopsis(eps)

    # Load episode objects for get_lines in MD writer
    episodes_by_num = {ep.num: ep for ep in eps}

    print(f"[rerank] Running global LLM ranking call...")
    ranked_promos, ranking_strategy = rank_promos_globally(
        promos,
        synopsis,
        llm,
        raw_output_path=str(out_dir / "raw" / "global_ranking_v2.json"),
    )
    print(f"[rerank] Ranking complete. Strategy: {ranking_strategy[:80]}...")

    # Print token summary
    print(f"[rerank] Tokens used — prompt: {llm.total_prompt_tokens}, completion: {llm.total_completion_tokens}")
    print(f"[rerank] Estimated cost: ${llm.estimate_cost():.4f}")

    # Write outputs
    write_promos_json(ranked_promos, str(out_dir / "promos.json"), ranking_strategy=ranking_strategy)
    write_promos_md(ranked_promos, episodes_by_num, str(out_dir / "promos.md"), ranking_strategy=ranking_strategy)
    print(f"[rerank] Done. Wrote {out_dir}/promos.json and {out_dir}/promos.md")


if __name__ == "__main__":
    main()
