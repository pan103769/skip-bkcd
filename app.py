import base64
import json
import os
from pathlib import Path
from typing import List

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b")
API_KEY = os.getenv("OLLAMA_API_KEY", "")

app = FastAPI(title="SkipSense Gemma Counter")

# Frontend and API are served from the same origin; CORS is not needed.
class CountRequest(BaseModel):
    frames: List[str] = Field(min_length=4, max_length=48)
    interval_ms: int = Field(default=250, ge=100, le=2000)

@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model": MODEL,
        "cloud_key_configured": bool(API_KEY),
    }

@app.post("/api/count")
def count_skips(payload: CountRequest):
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OLLAMA_API_KEY is not configured on the server yet."
        )

    try:
        image_bytes = []
        for data_url in payload.frames:
            encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
            image_bytes.append(base64.b64decode(encoded, validate=True))

        prompt = f"""
You are evaluating whether you can count jump-rope skips from ordered video frames.

The {len(image_bytes)} attached images are consecutive samples from ONE video,
in chronological order, sampled every {payload.interval_ms} milliseconds.
Count completed body jumps across the sequence. Use changes across frames, not
the number of frames. Do not count rope rotations separately. Do not invent
events that are not visible. The clip may start or end mid-jump.

Return ONLY valid JSON with these keys:
{{
  "completed_skips": integer or null,
  "confidence": "low" | "medium" | "high",
  "reason": "one short sentence describing visual evidence",
  "limitations": "one short sentence; mention if frame sampling makes exact counting uncertain"
}}
If an exact count cannot reasonably be determined, use null rather than pretending certainty.
"""
        client = ollama.Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=180.0,
        )
        response = client.chat(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": image_bytes,
            }],
            format="json",
            stream=False,
        )
        raw = response.message.content
        try:
            result = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            result = {
                "completed_skips": None,
                "confidence": "low",
                "reason": "Gemma did not return valid JSON.",
                "limitations": str(raw)[:500],
            }

        result["model"] = MODEL
        result["frames_analyzed"] = len(image_bytes)
        result["sampling_interval_ms"] = payload.interval_ms
        return result
    except HTTPException:
        raise
    except Exception as exc:
        # Keep this message useful without returning credentials or request data.
        raise HTTPException(
            status_code=502,
            detail=f"Ollama cloud request failed ({type(exc).__name__}). Check the model name and server logs."
        ) from exc
