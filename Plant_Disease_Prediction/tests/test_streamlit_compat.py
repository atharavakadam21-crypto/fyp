"""Static regression checks for the pinned Streamlit UI API."""
import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_st_image_calls_use_streamlit_122_compatible_width_argument():
    source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    image_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
        and node.func.attr == "image"
    ]

    assert image_calls, "Expected at least one st.image call"
    for call in image_calls:
        keyword_names = {keyword.arg for keyword in call.keywords}
        assert "use_container_width" not in keyword_names
        assert "use_column_width" in keyword_names


def test_production_and_candidate_checkpoints_are_not_modified_by_ui_fix():
    # UI compatibility tests intentionally don't load, retrain, or rewrite models.
    source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
    assert "trained_plant_disease_model.keras" in source
