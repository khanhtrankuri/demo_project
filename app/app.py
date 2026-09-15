from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.clip_encoder import CLIPEncoder
from src.query.query_parser import parse_query
from src.retrieval.hybrid_search import HybridSearchEngine, rerank_candidates
from src.retrieval.semantic_search import SemanticSearchEngine
from src.utils.config import load_config, resolve_path
from src.utils.feedback import append_feedback


PRESETS = [
    "pedestrian at night", "rainy road", "bus on city street", "cars on highway",
    "truck at night", "pedestrian on rainy road at night", "busy city street", "bicycle near cars",
]


@st.cache_resource(show_spinner="Loading local OpenCLIP model and FAISS index…")
def load_engine() -> tuple[SemanticSearchEngine, HybridSearchEngine, dict[str, Any]]:
    config = load_config()
    artifacts = resolve_path(config, config["paths"]["artifacts_dir"])
    model_cfg = config["model"]
    encoder = CLIPEncoder(
        model_cfg["name"], model_cfg["pretrained"],
        mixed_precision=bool(config["runtime"].get("mixed_precision", True)),
    )
    semantic = SemanticSearchEngine(encoder, artifacts / "indexes" / "all.faiss", artifacts / "mappings" / "all.json")
    hybrid = HybridSearchEngine(semantic, config["hybrid"]["semantic_weight"], config["hybrid"]["metadata_weight"])
    return semantic, hybrid, config


def store_feedback(query: str, item: dict[str, Any], relevant: bool, mode: str, config: dict[str, Any]) -> None:
    target = resolve_path(config, config["paths"]["processed_dir"]) / "feedback.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "query": query,
        "image_id": item["image_id"], "relevant": relevant, "search_mode": mode.lower().replace(" ", "_"),
        "clip_score": float(item.get("clip_score_raw", item.get("clip_score", 0))),
        "metadata_score": float(item.get("metadata_score", 0)), "final_score": float(item.get("final_score", 0)),
    }
    append_feedback(target, record)


def merged_requirements(query: str, weather: str, timeofday: str, scene: str, objects: list[str]) -> dict[str, Any]:
    parsed = parse_query(query)
    if weather != "Any": parsed["weather"] = weather
    if timeofday != "Any": parsed["timeofday"] = timeofday
    if scene != "Any": parsed["scene"] = scene
    if objects: parsed["objects"] = objects
    return parsed


st.set_page_config(page_title="RAV-14 SceneSearch", page_icon="🚘", layout="wide")
st.title("RAV-14 SceneSearch")
st.caption("Semantic Search for Autonomous Driving Scenes")

if "query" not in st.session_state:
    st.session_state.query = "pedestrian on a rainy city street at night"

st.write("Example queries")
preset_columns = st.columns(4)
for index, preset in enumerate(PRESETS):
    if preset_columns[index % 4].button(preset, key=f"preset-{index}", width="stretch"):
        st.session_state.query = preset

query = st.text_input("Query", key="query")
control_columns = st.columns([1.5, 1, 1, 1, 1])
mode = control_columns[0].selectbox("Search mode", ["Hybrid", "Semantic Only"])
top_k = control_columns[1].selectbox("Top-K", [5, 10, 20], index=1)
weather = control_columns[2].selectbox("Weather", ["Any", "clear", "overcast", "rainy", "snowy", "partly cloudy", "foggy"])
timeofday = control_columns[3].selectbox("Time of day", ["Any", "daytime", "nighttime", "dawn/dusk"])
scene = control_columns[4].selectbox("Scene", ["Any", "city street", "highway", "residential", "parking lot", "tunnel", "gas stations"])
object_filter = st.multiselect("Object", ["person", "car", "bus", "truck", "bike", "motorcycle", "traffic light", "traffic sign"])
hard_filter = st.checkbox("Hard metadata filtering", value=False, disabled=mode == "Semantic Only")

if st.button("SEARCH", type="primary", width="stretch"):
    if not query.strip():
        st.warning("Enter a driving-scene query.")
    else:
        try:
            semantic_engine, hybrid_engine, config = load_engine()
            requirements = merged_requirements(query, weather, timeofday, scene, object_filter)
            if mode == "Hybrid":
                results, info = hybrid_engine.search(
                    query, top_k=top_k, candidate_k=int(config["retrieval"]["candidate_k"]),
                    hard_filter=hard_filter, overrides=requirements,
                )
            else:
                candidates, latency = semantic_engine.search(query, top_k)
                results = rerank_candidates(candidates, requirements, alpha=1.0, beta=0.0, top_k=top_k)
                info = {"latency_ms": latency, "candidate_count": len(candidates), "mode": "semantic_only", "parsed_query": requirements}
            st.session_state.search_payload = {"query": query, "mode": mode, "results": results, "info": info}
        except Exception as exc:
            st.error(f"Search could not start: {exc}")

payload = st.session_state.get("search_payload")
if payload:
    info = payload["info"]
    st.caption(f"Mode: {payload['mode']} · Candidates: {info['candidate_count']} · Search latency: {info['latency_ms']:.2f} ms")
    results = payload["results"]
    if not results:
        st.info("No results matched the active hard filters.")
    else:
        _, _, feedback_config = load_engine()
        for start in range(0, len(results), 3):
            columns = st.columns(3)
            for column, item in zip(columns, results[start:start + 3]):
                with column:
                    st.subheader(f"#{item['rank']}")
                    st.image(item["image_path"], width="stretch")
                    st.write(f"Final Score: **{item.get('final_score', 0):.3f}**")
                    st.write(f"CLIP Score: `{item.get('clip_score_raw', item.get('clip_score', 0)):.3f}`")
                    st.write(f"Metadata Score: `{item.get('metadata_score', 0):.3f}`")
                    st.caption(f"Weather: {item.get('weather', '—')} · Time: {item.get('timeofday', '—')}")
                    st.caption(f"Scene: {item.get('scene', '—')}")
                    st.caption("Objects: " + (", ".join(item.get("objects", [])) or "—"))
                    feedback_columns = st.columns(2)
                    if feedback_columns[0].button("Relevant", key=f"yes-{item['image_id']}-{payload['query']}", width="stretch"):
                        store_feedback(payload["query"], item, True, payload["mode"], feedback_config)
                        st.toast("Feedback saved")
                    if feedback_columns[1].button("Not Relevant", key=f"no-{item['image_id']}-{payload['query']}", width="stretch"):
                        store_feedback(payload["query"], item, False, payload["mode"], feedback_config)
                        st.toast("Feedback saved")
