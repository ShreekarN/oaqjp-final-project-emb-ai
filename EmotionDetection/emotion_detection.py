
from typing import Dict, Optional, Any
import json
import requests

_WATSON_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
_HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}


def _search_for_key(obj: Any, key: str) -> Optional[float]:
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], (int, float)):
            return float(obj[key])
        for v in obj.values():
            res = _search_for_key(v, key)
            if res is not None:
                return res
    elif isinstance(obj, list):
        for item in obj:
            res = _search_for_key(item, key)
            if res is not None:
                return res
    return None


def _safe_parse_response(resp: requests.Response) -> Any:
    try:
        j = resp.json()
    except ValueError:
        try:
            j = json.loads(resp.text)
        except ValueError:
            return {}
    if isinstance(j, dict) and "text" in j and isinstance(j["text"], str):
        try:
            return json.loads(j["text"])
        except ValueError:
            return j
    return j


def _none_result() -> Dict[str, Optional[float]]:
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }


def emotion_detector(text_to_analyze: str) -> Dict[str, Optional[float]]:
    if not isinstance(text_to_analyze, str) or text_to_analyze.strip() == "":
        return _none_result()

    payload = {"raw_document": {"text": text_to_analyze}}
    try:
        resp = requests.post(_WATSON_URL, headers=_HEADERS, json=payload, timeout=10)
    except requests.RequestException:
        return _none_result()

    if resp.status_code == 400:
        return _none_result()

    parsed = _safe_parse_response(resp)
    results: Dict[str, Optional[float]] = {}
    for emotion in ("anger", "disgust", "fear", "joy", "sadness"):
        val = _search_for_key(parsed, emotion)
        results[emotion] = float(val) if val is not None else 0.0

    try:
        dominant = max(results.items(), key=lambda kv: kv[1])[0]
    except (ValueError, TypeError):
        dominant = None

    results["dominant_emotion"] = dominant
    return results
