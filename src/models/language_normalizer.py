from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from src.query.query_parser import parse_query


@dataclass
class NormalizedQuery:
    raw_query: str
    english_query: str
    source_language: str = "unknown"
    weather: str | None = None
    timeofday: str | None = None
    scene: str | None = None
    objects: list[str] | None = None
    model_output: str | None = None
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["objects"] = payload.get("objects") or []
        return payload


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def parse_language_model_output(text: str, raw_query: str) -> NormalizedQuery:
    data = _extract_json(text)
    fallback = parse_query(raw_query)
    english_query = str(data.get("english_query") or "").strip()
    if not english_query:
        return NormalizedQuery(
            raw_query=raw_query,
            english_query=raw_query.strip(),
            source_language=str(data.get("source_language") or "unknown"),
            weather=fallback.get("weather"),
            timeofday=fallback.get("timeofday"),
            scene=fallback.get("scene"),
            objects=list(fallback.get("objects") or []),
            model_output=text,
            fallback_used=True,
        )

    objects = data.get("objects") or []
    if isinstance(objects, str):
        objects = [objects]
    objects = [str(value).strip().lower() for value in objects if str(value).strip()]

    def optional_string(key: str) -> str | None:
        value = data.get(key)
        if value in (None, "", "null", "None"):
            return None
        return str(value).strip().lower()

    return NormalizedQuery(
        raw_query=raw_query,
        english_query=english_query,
        source_language=str(data.get("source_language") or "unknown").strip().lower(),
        weather=optional_string("weather"),
        timeofday=optional_string("timeofday"),
        scene=optional_string("scene"),
        objects=objects,
        model_output=text,
        fallback_used=False,
    )


class LanguageQueryNormalizer:
    """Normalize multilingual user prompts into a canonical English retrieval query.

    The model is intentionally isolated from CLIP/FAISS so it can be replaced later
    without changing the visual retrieval stack.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 160,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Multilingual query normalization requires transformers; install requirements.txt"
            ) from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

    @property
    def system_prompt(self) -> str:
        return (
            "You normalize multilingual video-search queries for a CLIP retrieval system. "
            "Translate the user's meaning into concise natural English. Preserve all visual details "
            "such as objects, actions, colors, weather, time of day, location, and spatial relations. "
            "Do not invent details. Return JSON only with keys: english_query, source_language, "
            "weather, timeofday, scene, objects. Use null for unknown scalar metadata and [] for "
            "unknown objects. The english_query must be a self-contained visual description suitable "
            "for text-to-image retrieval."
        )

    def normalize(self, query: str) -> NormalizedQuery:
        raw_query = query.strip()
        if not raw_query:
            raise ValueError("Query must not be empty")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": raw_query},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[:, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return parse_language_model_output(text, raw_query)
