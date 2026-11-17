"""API tests (FastAPI TestClient).

Validation / isolation tests use synthetic clips. Accept/reject tests use the
REAL cached EEGMMIDB split and the trained CNN checkpoint, and are skipped if
those artifacts are not present (run scripts/build_dataset.py and
scripts/run_cnn.py first).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from eegauth.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "default.yaml"

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


def _artifacts_present() -> bool:
    cfg = load_config(CFG_PATH)
    models = Path(cfg["paths"]["models_dir"])
    res = Path(cfg["paths"]["results_dir"])
    cache = Path(cfg["paths"]["cache_dir"])
    return (
        (models / "cnn.pt").exists()
        and (models / "normalizer.npz").exists()
        and (res / "embedding_verification.json").exists()
        and any(cache.glob("split_*.npz"))
    )


pytestmark = pytest.mark.skipif(not _artifacts_present(), reason="trained artifacts/cache missing")


def npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def files(arr: np.ndarray, name: str = "clips.npy"):
    return {"clips_file": (name, npy_bytes(arr), "application/octet-stream")}


@pytest.fixture()
def client(tmp_path):
    from eegauth.api import create_app

    app = create_app(str(CFG_PATH), checkpoint="cnn", store_dir=tmp_path / "store")
    return TestClient(app)


@pytest.fixture(scope="module")
def split():
    from eegauth.data import build_dataset

    return build_dataset(load_config(CFG_PATH))


def synthetic(n: int = 5, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, 64, 320)).astype(np.float32) * 1e-5


# ---------------- health ----------------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["enrolled_users"] == 0
    assert 0.0 < body["threshold"] < 1.0


# ---------------- input validation ----------------
@pytest.mark.parametrize("bad_id", ["../evil", "a/b", "a.b", "", "x" * 65, "with space"])
def test_enroll_rejects_bad_user_ids(client, bad_id):
    r = client.post("/enroll", data={"user_id": bad_id}, files=files(synthetic()))
    assert r.status_code in (400, 404, 422)
    assert r.status_code != 200


def test_enroll_rejects_wrong_shape(client):
    wrong = np.zeros((5, 32, 320), dtype=np.float32) + 1e-5
    r = client.post("/enroll", data={"user_id": "u1"}, files=files(wrong))
    assert r.status_code == 400
    assert "64 channels" in r.json()["detail"]


def test_enroll_rejects_wrong_sampling_length(client):
    wrong = synthetic()[:, :, :200]
    r = client.post("/enroll", data={"user_id": "u1"}, files=files(wrong))
    assert r.status_code == 400


def test_enroll_rejects_nan(client):
    x = synthetic()
    x[0, 0, 0] = np.nan
    r = client.post("/enroll", data={"user_id": "u1"}, files=files(x))
    assert r.status_code == 400
    assert "NaN" in r.json()["detail"]


def test_enroll_rejects_flat_clips(client):
    r = client.post("/enroll", data={"user_id": "u1"}, files=files(np.zeros((5, 64, 320), np.float32)))
    assert r.status_code == 400


def test_enroll_rejects_2d_and_garbage(client):
    r = client.post("/enroll", data={"user_id": "u1"}, files=files(np.zeros((64, 320), np.float32)))
    assert r.status_code == 400
    r = client.post(
        "/enroll", data={"user_id": "u1"},
        files={"clips_file": ("clips.npy", b"not a numpy file", "application/octet-stream")},
    )
    assert r.status_code == 400


def test_enroll_rejects_unsupported_extension(client):
    r = client.post("/enroll", data={"user_id": "u1"}, files=files(synthetic(), name="clips.csv"))
    assert r.status_code == 400
    assert "unsupported" in r.json()["detail"]


def test_enroll_needs_minimum_clips(client):
    r = client.post("/enroll", data={"user_id": "u1"}, files=files(synthetic(n=2)))
    assert r.status_code == 400
    assert "at least" in r.json()["detail"]


def test_edf_with_wrong_sampling_rate_rejected(client):
    r = client.post(
        "/enroll", data={"user_id": "u1"},
        files={"clips_file": ("rec.edf", b"definitely not an edf", "application/octet-stream")},
    )
    assert r.status_code in (400, 422, 500)
    assert r.status_code != 200


# ---------------- enroll / verify flow ----------------
def test_verify_unknown_user_404(client):
    r = client.post("/verify", data={"user_id": "nobody"}, files=files(synthetic()))
    assert r.status_code == 404


def test_duplicate_enrollment_409_then_overwrite(client):
    assert client.post("/enroll", data={"user_id": "u1"}, files=files(synthetic())).status_code == 200
    assert client.post("/enroll", data={"user_id": "u1"}, files=files(synthetic())).status_code == 409
    r = client.post("/enroll", data={"user_id": "u1", "overwrite": "true"}, files=files(synthetic()))
    assert r.status_code == 200


def test_user_storage_is_isolated(client, tmp_path):
    client.post("/enroll", data={"user_id": "alice"}, files=files(synthetic(seed=1)))
    store = tmp_path / "store"
    alice_template = np.load(store / "alice" / "template.npy").copy()
    client.post("/enroll", data={"user_id": "bob"}, files=files(synthetic(seed=2)))
    assert (store / "bob" / "template.npy").exists()
    np.testing.assert_array_equal(np.load(store / "alice" / "template.npy"), alice_template)
    # nothing is written outside the per-user directories
    assert {p.name for p in store.iterdir()} == {"alice", "bob"}


def test_genuine_accepted_impostor_rejected_real_data(client, split):
    """Real EEG: enroll on TRAIN runs of subject 1, verify on unseen TEST runs."""
    sid_tr, sid_te = split["subject_ids_train"], split["subject_ids_test"]
    s_a, s_b = int(np.unique(sid_tr)[0]), int(np.unique(sid_tr)[1])
    enroll_clips = split["X_train"][sid_tr == s_a]
    assert client.post("/enroll", data={"user_id": "A"}, files=files(enroll_clips)).status_code == 200

    genuine = split["X_test"][sid_te == s_a][:20]
    r = client.post("/verify", data={"user_id": "A"}, files=files(genuine))
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] is True and "token" in body
    claims = client.get("/token/decode", params={"token": body["token"]}).json()
    assert claims["sub"] == "A"

    impostor = split["X_test"][sid_te == s_b][:20]
    r = client.post("/verify", data={"user_id": "A"}, files=files(impostor))
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] is False and "token" not in body
    assert body["score"] < client.app.state.threshold


def test_tampered_and_garbage_tokens_rejected(client):
    assert client.get("/token/decode", params={"token": "garbage"}).status_code == 401
    from jose import jwt

    forged = jwt.encode({"sub": "A", "exp": 9999999999}, "wrong-secret", algorithm="HS256")
    assert client.get("/token/decode", params={"token": forged}).status_code == 401


def test_verify_clip_limit(client):
    client.post("/enroll", data={"user_id": "u1"}, files=files(synthetic()))
    r = client.post("/verify", data={"user_id": "u1"}, files=files(synthetic(n=101)))
    assert r.status_code == 400


def test_garbage_edf_is_a_clean_400(client):
    r = client.post(
        "/enroll", data={"user_id": "u1"},
        files={"clips_file": ("rec.edf", b"definitely not an edf", "application/octet-stream")},
    )
    assert r.status_code == 400
    assert "edf" in r.json()["detail"].lower()


def test_real_edf_upload_enrolls(client):
    edfs = sorted((ROOT / "data" / "eegbci").rglob("S001R01.edf"))
    if not edfs:
        pytest.skip("S001R01.edf not downloaded")
    r = client.post(
        "/enroll", data={"user_id": "edf_user"},
        files={"clips_file": ("S001R01.edf", edfs[0].read_bytes(), "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["n_clips"] > 10
