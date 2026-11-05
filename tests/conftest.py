import pytest

from app.core.config import get_settings

MODEL_PATH = get_settings().model_path

requires_model = pytest.mark.skipif(
    not MODEL_PATH.is_file(), reason="model missing – run scripts/download_model.py"
)
