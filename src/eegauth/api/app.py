"""FastAPI application: EEG-based login (enroll / verify / health).

Verification model: the Phase 3 CNN's pooled embedding (64-dim). Enrollment
stores the mean embedding ("template") of a user's clips; verification scores
a clip by cosine similarity to the template and accepts at the EER threshold
from results/embedding_verification.json. On accept, a JWT (30 min) is issued.

Security properties:
- Per-user storage isolation: each user's data lives under
  <store>/<user_id>/; user_id is strictly validated (no path traversal).
- Input validation: exact clip geometry (n, 64, 320), finite values, .npy or
  .edf uploads only.
- The JWT secret is a demo value from the config — NOT for production.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from jose import ExpiredSignatureError, JWTError, jwt

from eegauth.api.ingest import read_clips_upload, validate_user_id
from eegauth.models import EEGCNNSmall, get_device
from eegauth.train import PerChannelNormalizer

MIN_ENROLL_CLIPS = 3
MAX_VERIFY_CLIPS = 100


def create_app(
    config_path: str = "configs/default.yaml",
    checkpoint: str = "cnn",
    store_dir: Optional[Path] = None,
) -> FastAPI:
    """Build the app. checkpoint='cnn' (clean) or 'cnn_robust' (defended)."""
    from eegauth.config import load_config

    cfg = load_config(config_path)
    app = FastAPI(title="EEG-Auth login demo", version="0.1.0")
    device = get_device()

    ckpt = torch.load(
        Path(cfg["paths"]["models_dir"]) / f"{checkpoint}.pt",
        map_location=device, weights_only=False,
    )
    model = EEGCNNSmall(**ckpt["meta"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    normalizer = PerChannelNormalizer.load(Path(cfg["paths"]["models_dir"]) / "normalizer.npz")

    thr_path = Path(cfg["paths"]["results_dir"]) / "embedding_verification.json"
    thresholds = json.loads(thr_path.read_text(encoding="utf-8"))
    if checkpoint not in thresholds:
        raise RuntimeError(
            f"no embedding threshold for checkpoint '{checkpoint}' — run "
            f"scripts/eval_embeddings.py --checkpoint {checkpoint} first"
        )
    threshold = float(thresholds[checkpoint]["eer_threshold"])

    store = Path(store_dir) if store_dir else Path(cfg["paths"]["data_dir"]) / "store"
    store.mkdir(parents=True, exist_ok=True)

    app.state.cfg = cfg
    app.state.model = model
    app.state.normalizer = normalizer
    app.state.threshold = threshold
    app.state.store = store
    app.state.device = device
    app.state.checkpoint = checkpoint
    app.state.secret = cfg["api"]["secret_key"]
    app.state.token_minutes = int(cfg["api"]["token_expire_minutes"])

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    def user_dir(user_id: str) -> Path:
        return app.state.store / validate_user_id(user_id)

    def load_template(user_id: str) -> np.ndarray:
        path = user_dir(user_id) / "template.npy"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"user '{user_id}' is not enrolled")
        return np.load(path)

    def embed_clips(clips: np.ndarray) -> np.ndarray:
        xb = torch.as_tensor(app.state.normalizer.transform(clips), device=app.state.device)
        with torch.no_grad():
            emb = app.state.model.features(xb).cpu().numpy()
        return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

    def cosine_to_template(emb: np.ndarray, template: np.ndarray) -> float:
        t = template / max(float(np.linalg.norm(template)), 1e-12)
        return float(np.mean(emb @ t))

    def make_token(user_id: str) -> str:
        now = datetime.now(timezone.utc)
        claims = {
            "sub": user_id,
            "iat": now,
            "exp": now + timedelta(minutes=app.state.token_minutes),
            "scope": "eeg-login",
        }
        return jwt.encode(claims, app.state.secret, algorithm=cfg["api"]["jwt_algorithm"])

    @app.get("/health")
    def health() -> dict:
        n_users = len([p for p in app.state.store.iterdir() if p.is_dir()])
        return {
            "status": "ok",
            "model": app.state.checkpoint,
            "threshold": app.state.threshold,
            "enrolled_users": n_users,
        }

    @app.post("/enroll")
    async def enroll(
        user_id: str = Form(...),
        clips_file: UploadFile = File(...),
        overwrite: bool = Form(False),
    ) -> dict:
        try:
            udir = user_dir(user_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if (udir / "template.npy").exists() and not overwrite:
            raise HTTPException(status_code=409, detail=f"user '{user_id}' already enrolled")
        clips = read_clips_upload(clips_file.filename, await clips_file.read())
        if len(clips) < MIN_ENROLL_CLIPS:
            raise HTTPException(
                status_code=400,
                detail=f"enrollment needs at least {MIN_ENROLL_CLIPS} clips; got {len(clips)}",
            )
        emb = embed_clips(clips)
        template = emb.mean(axis=0)
        udir.mkdir(parents=True, exist_ok=True)
        np.save(udir / "template.npy", template)
        np.save(udir / "embeddings.npy", emb)
        meta = {
            "user_id": user_id,
            "n_clips": int(len(clips)),
            "enrolled_at": datetime.now(timezone.utc).isoformat(),
            "model": app.state.checkpoint,
        }
        (udir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"user_id": user_id, "n_clips": int(len(clips)), "status": "enrolled"}

    @app.post("/verify")
    async def verify(
        user_id: str = Form(...),
        clips_file: UploadFile = File(...),
    ) -> dict:
        template = load_template(user_id)  # 404 if unknown
        clips = read_clips_upload(clips_file.filename, await clips_file.read())
        if len(clips) > MAX_VERIFY_CLIPS:
            raise HTTPException(status_code=400, detail=f"verify at most {MAX_VERIFY_CLIPS} clips")
        emb = embed_clips(clips)
        scores = np.array([cosine_to_template(e[None, :], template) for e in emb])
        score = float(scores.mean())
        accepted = bool(score >= app.state.threshold)
        resp: dict[str, Any] = {
            "user_id": user_id,
            "accepted": accepted,
            "score": round(score, 6),
            "threshold": app.state.threshold,
            "n_clips": int(len(clips)),
        }
        if accepted:
            resp["token"] = make_token(user_id)
        return resp

    @app.get("/token/decode")
    def decode(token: str) -> dict:
        """Diagnostic endpoint: validates a JWT and returns its claims."""
        try:
            claims = jwt.decode(
                token, app.state.secret,
                algorithms=[cfg["api"]["jwt_algorithm"]],
                options={"require": ["exp", "sub"]},
            )
        except ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="token expired")
        except JWTError as exc:
            raise HTTPException(status_code=401, detail=f"invalid token: {exc}")
        return claims

    return app
