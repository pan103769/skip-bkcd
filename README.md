# SkipSense — Render + Ollama Cloud experiment

SkipSense uses MediaPipe Pose and a temporal peak detector to estimate jump cycles from ordered video frames. Gemma remains a separate workout-analysis feature: it receives measured count, duration, pace, and rest time and interprets those statistics without inventing measurements.

## Deploy on Render

1. Create a GitHub repository named `skipsense`.
2. Upload the project files to the repository root: `app.py`, `index.html`, `requirements.txt`, `render.yaml`, and `README.md`.
3. On Render, choose **New → Blueprint** and connect the repository. Render will read `render.yaml` and create the web service.
4. In Render's service environment settings, set `OLLAMA_API_KEY` to your Ollama API key. Create/manage the key in Ollama's official account settings. Never put this key in `index.html`, GitHub, or a public message.
5. Confirm `OLLAMA_MODEL` is a model name available to your Ollama Cloud account. Default is `gemma4:31b`; if the account lists a different Gemma 4 cloud model name, change the variable to that exact name and redeploy.
6. Open the deployed `https://...onrender.com` URL on your phone, allow camera access, record a clip, and review the estimate. Then use **Analyze workout with Gemma** for feedback.

Official docs:
- Render FastAPI deploy: https://render.com/docs/deploy-fastapi
- Render environment variables: https://render.com/docs/configure-environment-variables
- Ollama API authentication: https://github.com/ollama/ollama/blob/main/docs/api/authentication.mdx

## Limitations and safety

- This is a test, not a validated fitness tracker. Sampling every 100 ms can miss very fast or irregular jumps.
- It is not a continuously updating counter; it analyzes a short capture after recording.
- The counter detects repeated vertical hip motion, not rope rotation. Bouncing without a rope can be counted, and rope turns can be missed when the pose is occluded.
- Confidence describes pose coverage and cycle regularity, not validated accuracy. Compare each result with a manual count.
- Each analysis uses cloud inference and may consume quota or incur charges depending on your Ollama plan.
- Keep the API key secret. The key is read only by the server.
- Do not expose this as a public service without adding rate limits and request-size limits. The current prototype is for a small private demo.
- Render free services can spin down after inactivity, so the first request after a pause may take longer.

## Local setup

Use Python 3.11 for the MediaPipe dependency:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Camera access works on localhost or HTTPS. `OLLAMA_API_KEY` is only needed for the Gemma coach; set it in the server environment, never in the browser.

For local Gemma coaching, set the key for the current PowerShell session before starting the server:

```powershell
$env:OLLAMA_API_KEY = "paste-your-key-here"
$env:OLLAMA_MODEL = "gemma4:31b"
```

Do not commit a `.env` file or put the key in `index.html`.

To temporarily inspect decoded upload frames, set the debug variables in the same PowerShell window that starts Uvicorn:

```powershell
$env:SKIPSENSE_DEBUG_FRAMES = "1"
$env:SKIPSENSE_DEBUG_DIR = (Join-Path (Get-Location) "debug_frames")
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Startup prints the parsed debug state and output directory. `/api/health` also reports them. After uploading a video, inspect the first, middle, and last files in `debug_frames`. Disable the option with `Remove-Item Env:SKIPSENSE_DEBUG_FRAMES, Env:SKIPSENSE_DEBUG_DIR`.

## Demo checklist

1. Stand far enough back that the full body and both shoulders are visible.
2. Start the camera, record eight seconds, and compare the estimate with a manual count.
3. Check duration, pace, pose coverage, confidence, and limitations before asking Gemma for feedback.
4. Treat the result as a pose-based estimate, not proof that a rope rotated.

## Live Workout

Live Workout keeps a server-side session separate from upload analysis. The browser sends one camera sample about every 200 ms, or 5 FPS, instead of sending every camera frame. This keeps bandwidth and CPU practical; the trade-off is up to about 200 ms of counting/overlay latency. Start the camera, press **Start workout**, and use **Pause**, **Resume**, or **Reset**. MediaPipe landmarks are returned for the green skeleton overlay. Test the camera flow manually on localhost or HTTPS because automated tests cannot grant browser camera permission.

## Troubleshooting

- **MediaPipe import error:** confirm the virtual environment reports Python 3.11. Remove a Python 3.14 environment and recreate it with `py -3.11 -m venv .venv`.
- **Camera permission error:** use `http://127.0.0.1:8000` locally or the HTTPS Render URL, allow camera access, and ensure another application is not using the camera.
- **Low pose coverage:** move the camera farther away, improve lighting, and keep the complete body, hips, and shoulders in frame.
- **Diagnosing a failed upload:** keep the Uvicorn PowerShell window visible while retrying. Each count logs decoded frame count, detected-pose count, missing-pose count, visibility rejections, image size, and the RGB conversion path. No image data or API keys are logged.
- **Gemma returns 503:** set `OLLAMA_API_KEY` in the server environment. The counter does not require Gemma to estimate jumps.
- **Gemma returns 502:** verify the configured Gemma model name is available to the Ollama Cloud account and check the server logs for the upstream error type.
- **Render health check fails:** confirm the service starts with `uvicorn app:app --host 0.0.0.0 --port $PORT` and that `/api/health` returns JSON.

## Video sampling cost

Uploads can be sampled at 15, 30, or 60 frames per second, with a maximum of 2,000 frames. A 22.37-second video produces about 336, 672, or 1,343 frames respectively and is analyzed across the full original duration. A 30-second video at 60 FPS produces 1,800 frames. Higher rates create larger requests and require proportionally more MediaPipe CPU time; 60 FPS can be several times slower than 15 FPS. Videos requiring more than 2,000 frames use a lower effective sampling rate across their full duration, which is displayed in the results.
