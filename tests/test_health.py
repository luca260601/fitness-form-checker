from fastapi.testclient import TestClient
from app.pose_service.main import app

def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_analyze_stub(tmp_path):
    client = TestClient(app)
    dummy = tmp_path / "dummy.mp4"
    dummy.write_bytes(b"0000")  # Platzhalter
    with dummy.open("rb") as f:
        files = {"file": ("dummy.mp4", f, "video/mp4")}
        r = client.post("/analyze_squat", files=files)
    assert r.status_code == 200
    data = r.json()
    assert "angles" in data and "flags" in data
