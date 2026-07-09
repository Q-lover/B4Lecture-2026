from pathlib import Path


def test_runtime_source_does_not_import_reference_packages() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    forbidden = (
        "import normflows",
        "from normflows",
        "import glow_tts",
        "from glow_tts",
    )
    for path in src_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in text, f"{path} contains {marker!r}"
