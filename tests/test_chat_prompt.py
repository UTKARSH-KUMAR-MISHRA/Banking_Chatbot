from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("banking_app", ROOT / "app.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_build_multimodal_payload_contains_text_and_media_parts():
    payload = module.build_multimodal_payload(
        "Need account opening help",
        image_bytes=b"img",
        audio_bytes=b"audio",
        audio_mime_type="audio/wav",
    )

    assert len(payload) == 3
    assert any(getattr(part, "text", None) for part in payload if hasattr(part, "text"))
