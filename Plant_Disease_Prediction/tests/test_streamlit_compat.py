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


def test_improved_checkpoint_is_the_default_and_models_are_not_modified():
    # The app selects Model 3 by default; it never retrains or rewrites checkpoints.
    source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
    assert 'os.getenv("PLANTVISION_MODEL_PATH", "trained_plant_disease_model_improved.keras")' in source
    assert 'os.getenv("PLANTVISION_MODEL_VERSION", "improved")' in source
    assert "load_model(MODEL_PATH" in source
    assert "model.fit(" not in source


def test_runtime_pins_keep_streamlit_and_protobuf_compatible():
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    pins = {
        line.split("==", 1)[0]: line.split("==", 1)[1]
        for line in requirements.splitlines()
        if "==" in line and not line.lstrip().startswith("#")
    }
    assert pins.get("streamlit") == "1.22.0"
    assert pins.get("protobuf") == "3.20.3"
    assert pins.get("tensorflow") == "2.15.0"



def test_model_path_is_resolved_relative_to_main_file_not_terminal_directory():
    source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
    assert 'BASE_DIR = os.path.dirname(os.path.abspath(__file__))' in source
    assert 'os.path.join(BASE_DIR, _configured_model_path)' in source
    assert 'loaded_model.output_shape' in source
