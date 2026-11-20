import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from src.llm import LLMClient
from src.validator import ValidatedPromo


def rank_promos_globally(
    promos: List[ValidatedPromo],
    show_synopsis: str,
    llm: LLMClient,
    prompt_path: str = "prompts/rank_promos_v1.txt",
    raw_output_path: Optional[str] = None,
) -> Tuple[List[ValidatedPromo], str]:
    """
    Ranks validated promo candidates using a global LLM call.
    Considers hook strength, standalone clarity, beat diversity, and app conversion potential.
    """
    if not promos:
        return [], "No promos to rank."

    if len(promos) == 1:
        promos[0].rank = 1
        return promos, "Single promo present."

    # Build ultra-compact candidate representation for ranking prompt
    # Flatten scores to individual keys to save tokens vs. a string representation
    compact_candidates = []
    for p in promos:
        compact_candidates.append({
            "ep": p.episode_num,
            "cid": p.candidate_id,
            "dur": f"{round(p.duration, 1)}s",
            "hook": p.hook[:80],
            "cliff": p.cliffhanger[:80],
            "H": p.scores.get("hook_strength", 5),
            "C": p.scores.get("cliffhanger_strength", 5),
            "S": p.scores.get("standalone_clarity", 5),
            "comp": round(p.composite_score, 2),
        })

    candidates_json_str = json.dumps(compact_candidates)

    # Condense show synopsis: keep only first 70 chars per logline to fit within TPM limits
    # ~50 episodes × 70 chars ≈ 875 tokens vs 2868 for full synopsis
    synopsis_lines = []
    for line in show_synopsis.splitlines():
        stripped = line.strip()
        if stripped:
            synopsis_lines.append(stripped[:70])
    compact_synopsis = "\n".join(synopsis_lines)

    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    user_content = template.format(
        show_synopsis=compact_synopsis,
        candidates_json=candidates_json_str,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a senior growth marketing strategist for vertical microdramas. Return strictly valid JSON.",
        },
        {"role": "user", "content": user_content},
    ]

    # v3: compact synopsis (70-char/line) + flat score keys
    cache_key = f"global_rank_v4_{len(promos)}promos_{llm.model}"
    ranking_strategy = ""

    orig_max_tokens = llm.max_tokens
    llm.max_tokens = 4000

    try:
        parsed_json, raw_text, usage, was_cached = llm.call_chat_completion(
            messages,
            cache_key=cache_key,
            force_json=True,
        )

        if raw_output_path:
            p_out = Path(raw_output_path)
            p_out.parent.mkdir(parents=True, exist_ok=True)
            with open(p_out, "w", encoding="utf-8") as f:
                f.write(raw_text)

        ranking_strategy = parsed_json.get("ranking_strategy", "")
        ranked_items = parsed_json.get("ranked_promos", [])

        # Map (ep, cand_id) -> promo
        promo_map: Dict[Tuple[int, int], ValidatedPromo] = {
            (p.episode_num, p.candidate_id): p for p in promos
        }

        ordered_promos: List[ValidatedPromo] = []
        seen_keys = set()

        for item in ranked_items:
            ep_n = item.get("episode_num") or item.get("ep")
            c_id = item.get("candidate_id") or item.get("cid") or 1
            key = (ep_n, c_id)

            if key in promo_map and key not in seen_keys:
                p = promo_map[key]
                p.ranking_reason = item.get("one_line_reason", "")
                ordered_promos.append(p)
                seen_keys.add(key)

        # Append any promos missed by the ranking call (ordered by composite score)
        remaining = [p for p in promos if (p.episode_num, p.candidate_id) not in seen_keys]
        remaining.sort(key=lambda x: x.composite_score, reverse=True)
        ordered_promos.extend(remaining)

        # Assign final 1-based ranks
        for rank_idx, p in enumerate(ordered_promos, start=1):
            p.rank = rank_idx

        return ordered_promos, ranking_strategy

    except Exception as e:
        print(f"[Ranker] Warning: Global ranking call failed ({e}). Falling back to composite scores.")
        promos.sort(key=lambda p: p.composite_score, reverse=True)
        for rank_idx, p in enumerate(promos, start=1):
            p.rank = rank_idx
            p.ranking_reason = f"Ranked #{rank_idx} by automated composite score ({p.composite_score:.2f})."
        return promos, "Automated composite score fallback."
    finally:
        llm.max_tokens = orig_max_tokens
