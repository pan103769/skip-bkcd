# SkipSense — Render + Ollama Cloud experiment

This prototype tests **Gemma itself** as a jump-rope counter using your phone's live front camera. It records 8 seconds of video as sampled frames, then sends the ordered images to Ollama Cloud for one count attempt.

## Deploy on Render

1. Create a GitHub repository named `skipsense`.
2. Upload these four files to the repository root: `app.py`, `index.html`, `requirements.txt`, and `render.yaml`.
3. On Render, choose **New → Blueprint** and connect the repository. Render will read `render.yaml` and create the web service.
4. In Render's service environment settings, set `OLLAMA_API_KEY` to your Ollama API key. Create/manage the key in Ollama's official account settings. Never put this key in `index.html`, GitHub, or a public message.
5. Confirm `OLLAMA_MODEL` is a model name available to your Ollama Cloud account. Default is `gemma4:31b`; if the account lists a different Gemma 4 cloud model name, change the variable to that exact name and redeploy.
6. Open the deployed `https://...onrender.com` URL on your phone, allow camera access, record a clip, and click **Ask Gemma to count**.

Official docs:
- Render FastAPI deploy: https://render.com/docs/deploy-fastapi
- Render environment variables: https://render.com/docs/configure-environment-variables
- Ollama API authentication: https://github.com/ollama/ollama/blob/main/docs/api/authentication.mdx

## Limitations and safety

- This is a test, not a validated fitness tracker. Sampling every 250 ms can miss fast jumps.
- It is not a continuously updating counter; it analyzes a short capture after recording.
- Each analysis uses cloud inference and may consume quota or incur charges depending on your Ollama plan.
- Keep the API key secret. The key is read only by the server.
- Do not expose this as a public service without adding rate limits and request-size limits. The current prototype is for a small private demo.
- Render free services can spin down after inactivity, so the first request after a pause may take longer.
