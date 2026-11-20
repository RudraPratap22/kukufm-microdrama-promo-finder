# Kuku Promo Finder

A CLI tool that reads microdrama screenplays and finds the best 20–60 second promo clips for Instagram, TikTok, and YouTube Shorts.

## What it does

1. **Parses** all screenplay `.txt` files from a show folder (pure Python, no LLM)
2. **Finds candidates** per episode via an LLM call: top 2–3 clips each with hook, cliffhanger, timestamps, and standalone-clarity scores
3. **Validates** every candidate: timestamp bounds, 20–60s duration, no overlaps; snaps to nearest dialogue line
4. **Re-ranks globally** (one final LLM call): weights hook strength, standalone clarity, and beat diversity — penalizes near-duplicate hooks and clips starting mid-scene with no context
5. **Writes output**: `promos.json` (machine-readable), `promos.md` (human-reviewable with verbatim screenplay lines for fast quality check), `validation_log.json`

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd "KukuFM assignemnt"

# 2. Install dependencies (Python 3.11+)
pip install -r requirements.txt

# 3. Create local config and configure API key
cp config.example.yaml config.yaml
cp .env.example .env
# Then open .env and add your key:
#   GROQ_API_KEY=gsk_...
```

> **Model is configured in your local `config.yaml`** — see "Swap model" below.

---

## Run on a show

```bash
python run.py --input "path/to/show/folder" --out output
```

The show folder must contain `.txt` files in the format `episode-NNN-*.txt`.

**Examples:**

```bash
# Full 50-episode run
python run.py --input "path/to/show folder" --out output

# Single episode (fast test)
python run.py --input "path/to/show folder" --out output --episodes 1

# Range of episodes
python run.py --input "path/to/show folder" --out output --episodes 1-5

# Force fresh LLM calls (skip disk cache)
python run.py --input "path/to/show folder" --out output --no-cache
```

A partial run **resumes automatically** — already-cached episodes are skipped, so you never re-spend tokens.

---

## Swap model (one-line change)

Edit `config.yaml`:

```yaml
model: "qwen/qwen3.8-27b"   # <- change this line only
```

Any OpenAI-compatible endpoint works. Update `base_url` and `api_key_env` as needed. No other code changes required.

**Tested models (Groq free tier):**
| Model | Quality | Speed | Notes |
|---|---|---|---|
| `openai/gpt-oss-120b` | ★★★★★ | Slower | Reasoning model; set `max_tokens: 6000` |
| `qwen/qwen3.8-27b` | ★★★★☆ | Fast | May occasionally declare "no viable promo" for weaker episodes |
| `openai/gpt-oss-20b` | ★★★★☆ | Medium | Smaller reasoning model |

---

## Include camera blocks

By default, `CAMERA:` blocks are omitted from prompts (saves ~45% tokens, scores are identical).

```bash
python run.py --input "..." --out output --include-camera
```

> ⚠️ Camera blocks push per-episode prompts from ~5,200 to ~7,400 tokens. On Groq free tier (8,000 TPM), episodes longer than ~130s will hit rate limits.

---

## Output files

| File | Purpose |
|---|---|
| `output/promos.json` | Machine-readable: rank, episode, segments with timestamps, hook, cliffhanger, scores |
| `output/promos.md` | Human-reviewable: verbatim screenplay lines for each segment, quality checklist row |
| `output/validation_log.json` | Rejection reasons (bad timestamps, duration out of range, overlaps) |
| `output/raw/episode_NNN_<model>_<model-hash>_find_candidates_v1_<cam-or-nocam>_candidates.json` | Raw LLM response per episode and inference configuration |
| `output/raw/global_ranking.json` | Raw global ranking LLM response |

**Quality check workflow** (`promos.md`):
Each promo entry ends with:
```
Hook [ ]  Cliffhanger [ ]  Standalone [ ]
```
Read the verbatim lines, watch the clip, tick the boxes. If you spot a bad pick, check `validation_log.json` and the raw LLM response for that episode.

---

## Re-rank without re-running all episodes

If you want to re-run only the global ranking step (e.g., to try a stronger model):

```bash
python rerank.py --input "path/to/show folder" --out output
```

This loads existing `output/promos.json` and rewrites it with fresh LLM rankings. All episode candidate calls stay cached.

---

## Cost per show (at scale)

| Provider | Model | Approx cost per 50-episode show |
|---|---|---|
| Groq (free tier) | `qwen/qwen3.8-27b` | ~\$0.15 |
| Groq (free tier) | `openai/gpt-oss-120b` | ~\$0.17 |
| OpenAI | `gpt-4o-mini` | ~\$0.04 |
| OpenAI | `gpt-4o` | ~\$1.30 |

Assumptions: 50 episodes × ~5,200 avg prompt tokens = ~260,000 prompt tokens. Completion tokens ~15,000. Disk caching means reruns (same show, same model) cost \$0.

---

## Known limitations (v1)

- **Single-episode clips only**: Promo segments are from one episode at a time. Cross-episode joins (e.g., the hook from Ep 3 + the cliffhanger from Ep 7) are out of scope for v1.
- **Timestamp validation and snapping**: Every proposed timestamp must match a timestamp in the screenplay. Start times snap to the nearest dialogue or scene start; end times snap to the nearest block end.
- **Free-tier rate limits**: Groq on-demand free tier is 8,000 tokens/minute. The pipeline retries with exponential backoff, but a full 50-episode run takes 15–30 minutes due to rate-limiting pauses. A paid tier (or OpenAI) removes this bottleneck.
- **LLM false negatives**: Some shorter or simpler episodes may be declared "no viable promo" by weaker models. Check `output/validation_log.json` for the count.

---

## Project structure

```
run.py                   # Main CLI entrypoint
rerank.py                # Standalone re-ranker (skips episode calls)
config.example.yaml      # Safe provider/model template; copy to config.yaml
config.yaml              # Local configuration (ignored by Git)
.env                     # API keys — never committed
requirements.txt         # httpx, pyyaml, python-dotenv
prompts/
  find_candidates_v1.txt # Per-episode candidate extraction prompt
  rank_promos_v1.txt     # Global ranking prompt
src/
  models.py              # Block, Scene, Episode dataclasses
  parser.py              # parse_screenplay_text(), get_lines(), format_episode_for_llm()
  context.py             # build_show_synopsis(), get_episode_context()
  llm.py                 # LLMClient: cache, retries, backoff, token counting
  validator.py           # CandidateValidator: timestamp/duration/overlap checks + snapping
  ranker.py              # rank_promos_globally(): global LLM ranking call
  output_writer.py       # write_promos_json(), write_promos_md()
  validate_dataset.py    # Standalone dataset validation summary
```
