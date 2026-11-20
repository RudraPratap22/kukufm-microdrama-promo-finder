import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import httpx
import yaml
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Estimated costs per 1 Million tokens (USD)
MODEL_PRICING = {
    # Groq models
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    # OpenAI models
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


class LLMClient:
    def __init__(
        self,
        config_path: str = "config.yaml",
        model_override: Optional[str] = None,
        cache_dir: str = ".cache/llm",
    ):
        self.config = self._load_config(config_path)
        if model_override:
            self.config["model"] = model_override

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        api_key_env = self.config.get("api_key_env", "GROQ_API_KEY")
        self.api_key = os.environ.get(api_key_env, "").strip()

        self.base_url = self.config.get("base_url", "https://api.groq.com/openai/v1").rstrip("/")
        self.model = self.config.get("model", "llama-3.3-70b-versatile")
        self.temperature = float(self.config.get("temperature", 0.1))
        self.max_tokens = int(self.config.get("max_tokens", 3000))
        self.timeout = float(self.config.get("request_timeout", 60.0))
        self.max_retries = int(self.config.get("max_retries", 5))
        self.retry_base_delay = float(self.config.get("retry_base_delay", 2.0))

        # Session tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.cache_hits = 0

    def _load_config(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _get_cache_path(self, cache_key: str) -> Path:
        hashed = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{hashed}.json"

    def estimate_cost(self) -> float:
        pricing = MODEL_PRICING.get(self.model, {"input": 0.50, "output": 0.80})
        cost = (self.total_prompt_tokens / 1_000_000.0) * pricing["input"] + (
            self.total_completion_tokens / 1_000_000.0
        ) * pricing["output"]
        return cost

    def _clean_json_text(self, text: str) -> str:
        """Strip markdown markers and find outermost JSON braces."""
        text = text.strip()
        if text.startswith("```"):
            # Strip ```json ... ```
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Find first { and last }
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace : last_brace + 1]
        return text

    def call_chat_completion(
        self,
        messages: List[Dict[str, str]],
        cache_key: Optional[str] = None,
        force_json: bool = True,
    ) -> Tuple[Dict[str, Any], str, Dict[str, int], bool]:
        """
        Executes chat completion with caching, exponential backoff, and JSON parsing.
        Returns: (parsed_json_dict, raw_text, usage_dict, was_cache_hit)
        """
        # 1. Check disk cache
        if cache_key:
            cache_file = self._get_cache_path(cache_key)
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    self.cache_hits += 1
                    usage = cached.get("usage", {})
                    p_tok = usage.get("prompt_tokens", 0)
                    c_tok = usage.get("completion_tokens", 0)
                    self.total_prompt_tokens += p_tok
                    self.total_completion_tokens += c_tok
                    self.total_calls += 1
                    return cached["parsed"], cached["raw"], usage, True
                except Exception as e:
                    # Corrupted cache, continue with live call
                    pass

        # 2. Check API key
        if not self.api_key:
            raise ValueError(
                f"API Key is missing! Set it in your .env file or environment variable '{self.config.get('api_key_env', 'GROQ_API_KEY')}'"
            )

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if force_json:
            payload["response_format"] = {"type": "json_object"}

        # 3. Live call with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(endpoint, headers=headers, json=payload)

                if resp.status_code == 200:
                    data = resp.json()
                    choice = data["choices"][0]
                    raw_content = choice["message"]["content"]
                    usage = data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

                    p_tok = usage.get("prompt_tokens", 0)
                    c_tok = usage.get("completion_tokens", 0)
                    self.total_prompt_tokens += p_tok
                    self.total_completion_tokens += c_tok
                    self.total_calls += 1

                    # Parse JSON
                    parsed_json = self._parse_json_defensive(raw_content)

                    # Save to cache
                    if cache_key:
                        cache_file = self._get_cache_path(cache_key)
                        cache_data = {
                            "cache_key": cache_key,
                            "model": self.model,
                            "timestamp": time.time(),
                            "usage": usage,
                            "raw": raw_content,
                            "parsed": parsed_json,
                        }
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(cache_data, f, indent=2)

                    return parsed_json, raw_content, usage, False

                elif resp.status_code in (429, 500, 502, 503, 504):
                    # Rate limit or temporary server error
                    sleep_time = (self.retry_base_delay * (2**attempt)) + random.uniform(0.5, 1.5)
                    # Check for explicit retry-after header
                    retry_hdr = resp.headers.get("retry-after")
                    if retry_hdr:
                        try:
                            sleep_time = float(retry_hdr) + 1.0
                        except ValueError:
                            pass
                    else:
                        # Extract "Please try again in Xs" from body if present
                        m = re.search(r"try again in ([0-9.]+)s", resp.text)
                        if m:
                            sleep_time = float(m.group(1)) + 1.5

                    if sleep_time > 120:
                        print(
                            f"[LLM] Provider rate-limit window reset required: {sleep_time:.1f}s ({sleep_time/60:.1f} minutes). Pausing execution until reset..."
                        )
                    else:
                        print(
                            f"[LLM] Status {resp.status_code} received. Sleeping {sleep_time:.2f}s before retry (attempt {attempt + 1}/{self.max_retries})..."
                        )
                    time.sleep(sleep_time)
                    last_error = f"HTTP {resp.status_code}: {resp.text}"
                else:
                    raise RuntimeError(f"LLM API Error (HTTP {resp.status_code}): {resp.text}")

            except httpx.RequestError as e:
                sleep_time = (self.retry_base_delay * (2**attempt)) + random.uniform(0.1, 1.0)
                print(f"[LLM] Connection error: {e}. Retrying in {sleep_time:.2f}s (attempt {attempt + 1}/{self.max_retries})...")
                time.sleep(sleep_time)
                last_error = str(e)

        raise RuntimeError(f"Exceeded max retries ({self.max_retries}). Last error: {last_error}")

    def _parse_json_defensive(self, raw_text: str) -> Dict[str, Any]:
        """Try normal JSON parse, fallback cleanup, and if needed one repair call."""
        cleaned = self._clean_json_text(raw_text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Attempt a single repair call
            print("[LLM] Malformed JSON received, attempting one-time repair...")
            try:
                repair_messages = [
                    {
                        "role": "system",
                        "content": "You are a JSON repair tool. Output only valid, parseable JSON matching the user data. No explanations.",
                    },
                    {
                        "role": "user",
                        "content": f"Fix and return only the valid JSON for this malformed text:\n\n{raw_text}",
                    },
                ]
                endpoint = f"{self.base_url}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.model,
                    "messages": repair_messages,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                }
                with httpx.Client(timeout=30.0) as client:
                    repair_resp = client.post(endpoint, headers=headers, json=payload)
                if repair_resp.status_code == 200:
                    repair_content = repair_resp.json()["choices"][0]["message"]["content"]
                    return json.loads(self._clean_json_text(repair_content))
            except Exception as e:
                pass
            # If repair also fails, raise original error
            return json.loads(cleaned)
