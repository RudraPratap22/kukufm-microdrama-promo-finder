import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from src.context import build_show_synopsis, get_episode_context
from src.llm import LLMClient
from src.models import Episode
from src.output_writer import write_promos_json, write_promos_md
from src.parser import format_episode_for_llm, parse_episode_file
from src.ranker import rank_promos_globally
from src.validator import CandidateValidator, ValidatedPromo

CANDIDATE_PROMPT_VERSION = "find_candidates_v1"


def candidate_raw_path(raw_dir: Path, episode_num: int, model: str, include_camera: bool) -> Path:
    model_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
    model_hash = hashlib.sha256(model.encode("utf-8")).hexdigest()[:10]
    camera_tag = "cam" if include_camera else "nocam"
    return raw_dir / f"episode_{episode_num:03d}_{model_tag}_{model_hash}_{CANDIDATE_PROMPT_VERSION}_{camera_tag}_candidates.json"


def parse_episodes_arg(ep_arg: Optional[str], all_nums: List[int]) -> List[int]:
    """Parse episode range or list: '1', '1-5', '1,2,5', or None for all."""
    if not ep_arg:
        return sorted(all_nums)

    selected = set()
    parts = ep_arg.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start_i, end_i = int(start_s), int(end_s)
            selected.update(range(start_i, end_i + 1))
        elif part.isdigit():
            selected.add(int(part))

    return sorted([num for num in selected if num in all_nums])


def load_all_episodes(input_dir: str) -> Dict[int, Episode]:
    folder = Path(input_dir)
    files = sorted(folder.glob("episode-*.txt"))
    if not files:
        raise FileNotFoundError(f"No episode-*.txt files found in: {input_dir}")

    episodes_by_num: Dict[int, Episode] = {}
    for f in files:
        ep = parse_episode_file(f)
        episodes_by_num[ep.num] = ep
    return episodes_by_num


def load_prompt_template(prompt_path: str = "prompts/find_candidates_v1.txt") -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="Microdrama Promo Clip Extractor & Ranker")
    parser.add_argument("--input", required=True, help="Path to folder containing episode-NNN-*.txt scripts")
    parser.add_argument("--out", default="output", help="Output directory (default: output)")
    parser.add_argument("--episodes", default=None, help="Episodes to process: e.g. '1', '1-5', '1,3,7'")
    parser.add_argument("--model", default=None, help="Override model name in config.yaml")
    parser.add_argument("--include-camera", action="store_true", help="Include camera blocks in screenplay text sent to LLM")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--no-cache", action="store_true", help="Bypass disk cache for fresh LLM inference")

    args = parser.parse_args()

    # 1. Load episodes
    print(f"Loading scripts from: {args.input}")
    episodes_by_num = load_all_episodes(args.input)
    all_ep_nums = sorted(list(episodes_by_num.keys()))
    selected_nums = parse_episodes_arg(args.episodes, all_ep_nums)

    print(f"Loaded {len(episodes_by_num)} total episodes. Selected {len(selected_nums)} episodes: {selected_nums}")

    # 2. Build full show synopsis
    show_synopsis = build_show_synopsis(list(episodes_by_num.values()))

    # 3. Initialize LLM Client and Validator
    llm = LLMClient(config_path=args.config, model_override=args.model)
    validator = CandidateValidator()
    prompt_template = load_prompt_template(f"prompts/{CANDIDATE_PROMPT_VERSION}.txt")

    out_dir = Path(args.out)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    promos_json_path = out_dir / "promos.json"
    promos_md_path = out_dir / "promos.md"
    validation_log_path = out_dir / "validation_log.json"

    all_validated_promos: List[ValidatedPromo] = []

    print(f"\nUsing Model: {llm.model} (Base URL: {llm.base_url})")
    print(f"Camera Blocks Included: {args.include_camera}\n")

    # 4. Process each selected episode
    cam_flag = "cam" if args.include_camera else "nocam"
    for ep_num in selected_nums:
        ep = episodes_by_num[ep_num]
        raw_path = candidate_raw_path(raw_dir, ep_num, llm.model, args.include_camera)

        # Check if raw candidate JSON already exists on disk (resume capability)
        if raw_path.exists() and not args.no_cache:
            try:
                with open(raw_path, "r", encoding="utf-8") as f:
                    parsed_json = json.load(f)
                promos = validator.validate_episode_candidates(ep, parsed_json)
                print(f"--- Episode {ep_num:02d} ({ep.duration:.1f}s) --- [RESUMED FROM DISK] ({len(promos)} valid candidate(s))")
                all_validated_promos.extend(promos)
                continue
            except Exception:
                pass

        print(f"--- Processing Episode {ep_num:02d} ({ep.duration:.1f}s) ---")

        # Prepare context and screenplay text
        ctx = get_episode_context(ep, episodes_by_num, show_synopsis)
        screenplay_text = format_episode_for_llm(ep, include_camera=args.include_camera)

        user_content = prompt_template.format(
            show_synopsis=ctx["show_synopsis"],
            prev_episode_logline=ctx["prev_episode_logline"],
            episode_logline=ctx["episode_logline"],
            continuity_notes=ctx["continuity_notes"],
            screenplay_text=screenplay_text,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a professional social-media video editor specializing in viral vertical microdramas. Return strictly valid JSON.",
            },
            {"role": "user", "content": user_content},
        ]

        cache_key = None if args.no_cache else f"{CANDIDATE_PROMPT_VERSION}_ep{ep_num:03d}_{llm.model}_{cam_flag}"

        try:
            parsed_json, raw_text, usage, was_cached = llm.call_chat_completion(
                messages,
                cache_key=cache_key,
                force_json=True,
            )
            status_str = "CACHE HIT" if was_cached else "LIVE CALL"
            p_tok = usage.get("prompt_tokens", 0)
            c_tok = usage.get("completion_tokens", 0)
            print(f"  [LLM {status_str}] Prompt: {p_tok:,} tokens, Output: {c_tok:,} tokens")

            # Save raw response
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(raw_text)

            # Validate candidates
            promos = validator.validate_episode_candidates(ep, parsed_json)
            print(f"  [Validator] {len(promos)} valid candidate(s) accepted (raw proposed: {len(parsed_json.get('candidates', []))})")
            all_validated_promos.extend(promos)

            # Incremental save
            write_promos_json(all_validated_promos, str(promos_json_path))
            write_promos_md(all_validated_promos, episodes_by_num, str(promos_md_path))
            validator.write_validation_log(str(validation_log_path))

            if not was_cached:
                time.sleep(1.5)

        except Exception as e:
            print(f"  [ERROR] Failed processing episode {ep_num}: {e}")
            validator.log_rejection(ep_num, "EXCEPTION", str(e))

    # 5. Global Ranking (Step 5)
    print(f"\n--- Global Promo Ranking ({len(all_validated_promos)} candidate promos) ---")
    raw_rank_path = raw_dir / "global_ranking.json"
    ranked_promos, ranking_strategy = rank_promos_globally(
        promos=all_validated_promos,
        show_synopsis=show_synopsis,
        llm=llm,
        raw_output_path=str(raw_rank_path),
    )
    if ranking_strategy:
        print(f"  [Ranker Strategy] {ranking_strategy}")

    # 6. Write outputs
    promos_json_path = out_dir / "promos.json"
    promos_md_path = out_dir / "promos.md"
    validation_log_path = out_dir / "validation_log.json"

    write_promos_json(ranked_promos, str(promos_json_path))
    write_promos_md(ranked_promos, episodes_by_num, str(promos_md_path), ranking_strategy=ranking_strategy)
    validator.write_validation_log(str(validation_log_path))

    cost = llm.estimate_cost()
    print("\n" + "=" * 60)
    print("Execution Finished Summary")
    print("=" * 60)
    print(f"Episodes Processed: {len(selected_nums)}")
    print(f"Total Validated Promos: {len(all_validated_promos)}")
    print(f"Rejections Logged: {len(validator.rejection_log)}")
    print(f"Total Prompt Tokens: {llm.total_prompt_tokens:,}")
    print(f"Total Completion Tokens: {llm.total_completion_tokens:,}")
    print(f"Total LLM Calls: {llm.total_calls} ({llm.cache_hits} cache hits)")
    print(f"Estimated Cost: ${cost:.5f} USD")
    print(f"Outputs written to: {out_dir}")
    print(f"  - Machine-readable: {promos_json_path}")
    print(f"  - Human-reviewable: {promos_md_path}")
    print(f"  - Validation log:   {validation_log_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
