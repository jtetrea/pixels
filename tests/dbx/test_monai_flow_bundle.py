from pathlib import Path

import pandas as pd
import pytest

from dbx.pixels.modelserving.bundles.monai_runtime import (
    MONAI_DEPLOY_PACKAGES,
    REQUIRED_NOTEBOOK_CONFIG,
    TRITON_CLIENT_PACKAGES,
    load_monai_notebook_config,
)
from dbx.pixels.modelserving.bundles.monai_flow import (
    MonaiDeployAppModel,
    MonaiFlowBundleTransformer,
    _build_prediction_payload,
    _log_pyfunc_model,
    _merge_requirements,
    _normalise_prediction_result,
    _relax_pipeline_generator_python_constraint,
    log_monai_deploy_app,
    log_monai_flow_bundle,
)


def test_build_prediction_payload_minimal():
    assert _build_prediction_payload("/Volumes/main/schema/vol/input.nii.gz") == {
        "image_path": "/Volumes/main/schema/vol/input.nii.gz"
    }


def test_build_prediction_payload_with_optional_fields():
    payload = _build_prediction_payload(
        "/Volumes/main/schema/vol/input.nii.gz",
        output_dir="/Volumes/main/schema/vol/output",
        label_prompt=[3, 4],
        modality="CT_BODY",
    )

    assert payload == {
        "image_path": "/Volumes/main/schema/vol/input.nii.gz",
        "output_dir": "/Volumes/main/schema/vol/output",
        "label_prompt": "3,4",
        "modality": "CT_BODY",
    }


def test_normalise_prediction_result_from_dict():
    result = _normalise_prediction_result(
        {
            "output_dir": "/tmp/out",
            "output_files": ("/tmp/out/a.nii.gz",),
            "labels": {1: "spleen"},
        }
    )

    assert result == {
        "output_dir": "/tmp/out",
        "output_files": ["/tmp/out/a.nii.gz"],
        "labels": {"1": "spleen"},
        "error": "",
    }


def test_normalise_prediction_result_from_dataframe():
    result = _normalise_prediction_result(
        pd.DataFrame(
            [
                {
                    "output_dir": "/tmp/out",
                    "output_files": ["/tmp/out/a.nii.gz"],
                    "labels": {"1": "spleen"},
                }
            ]
        )
    )

    assert result["output_dir"] == "/tmp/out"
    assert result["output_files"] == ["/tmp/out/a.nii.gz"]
    assert result["labels"] == {"1": "spleen"}
    assert result["error"] == ""


def test_normalise_prediction_result_from_dicom_output():
    result = _normalise_prediction_result(
        {
            "output_dir": "/Volumes/main/schema/vol/output",
            "output_files": "/Volumes/main/schema/vol/output/seg.dcm",
        }
    )

    assert result["output_files"] == ["/Volumes/main/schema/vol/output/seg.dcm"]
    assert result["error"] == ""


def test_merge_requirements_keeps_first_package_specifier():
    requirements = _merge_requirements(
        ["monai>=1.5", "pydicom>=2.3"],
        ["monai==1.5.0", "highdicom"],
    )

    assert requirements == ["monai>=1.5", "pydicom>=2.3", "highdicom"]


def test_runtime_requirements_override_generated_app_pins():
    from dbx.pixels.modelserving.bundles.monai_flow import (
        DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS,
    )

    requirements = _merge_requirements(
        DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS,
        ["monai==1.4.0", "torch==2.4.0", "monai-deploy-app-sdk>=3.0.0"],
    )

    assert "monai==1.5.1" in requirements
    assert "torch" in requirements
    assert "monai-deploy-app-sdk==3.5.0" in requirements
    assert "monai==1.4.0" not in requirements
    assert "torch==2.4.0" not in requirements
    assert "monai-deploy-app-sdk>=3.0.0" not in requirements


def test_deploy_app_model_normalizes_inputs():
    model = MonaiDeployAppModel()

    assert model._normalize_input("/Volumes/main/schema/vol/input") == {
        "image_path": "/Volumes/main/schema/vol/input"
    }
    assert model._normalize_input([{"image_path": "a", "output_dir": "b"}]) == {
        "image_path": "a",
        "output_dir": "b",
    }


def test_log_monai_flow_bundle_validates_deploy_app_before_logging():
    with pytest.raises(FileNotFoundError):
        log_monai_flow_bundle(deploy_app_dir="/missing/deploy/app")


def test_log_monai_deploy_app_aliases_legacy_helper():
    assert log_monai_deploy_app is log_monai_flow_bundle


def test_log_pyfunc_model_supports_mlflow_2_artifact_path():
    calls = []

    class FakePyfunc:
        def log_model(self, artifact_path, python_model=None):
            calls.append(
                {"artifact_path": artifact_path, "python_model": python_model}
            )

    class FakeMlflow:
        pyfunc = FakePyfunc()

    model = object()
    _log_pyfunc_model(FakeMlflow(), name="monai_deploy_app_model", python_model=model)

    assert calls == [
        {"artifact_path": "monai_deploy_app_model", "python_model": model}
    ]


def test_runtime_requirements_pin_non_breaking_holoscan():
    from dbx.pixels.modelserving.bundles.monai_flow import (
        DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS,
    )

    assert "wheel" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "filelock" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "wheel-axle-runtime<1.0" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "mlflow>=2.17,<3.0" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "databricks-connect>=15.4.2,<16" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "monai-deploy-app-sdk==3.5.0" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "holoscan==3.10.0" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "holoscan-cu12==3.10.0" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "protobuf>=5.26.1,<6.0dev" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "grpcio>=1.67.1,<1.68" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "grpcio-status>=1.67.1,<1.68" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "tritonclient[http,grpc]==2.60.0" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "lazy-loader>=0.4" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "python-utils>=3.8" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS


def test_notebook_runtime_packages_include_databricks_fixes():
    assert "monai==1.5.1" in MONAI_DEPLOY_PACKAGES
    assert "monai-deploy-app-sdk==3.5.0" in MONAI_DEPLOY_PACKAGES
    assert "holoscan-cu12==3.10.0" in MONAI_DEPLOY_PACKAGES
    assert "lazy-loader>=0.4" in MONAI_DEPLOY_PACKAGES
    assert "python-utils>=3.8" in MONAI_DEPLOY_PACKAGES
    assert "grpcio>=1.67.1,<1.68" in TRITON_CLIENT_PACKAGES
    assert "grpcio-status>=1.67.1,<1.68" in TRITON_CLIENT_PACKAGES
    assert "tritonclient[http,grpc]==2.60.0" in TRITON_CLIENT_PACKAGES


def test_notebook_config_requires_values(monkeypatch):
    for env_name in REQUIRED_NOTEBOOK_CONFIG.values():
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.delenv("PIXELS_MONAI_MODEL_ID", raising=False)

    with pytest.raises(ValueError, match="Notebook 9 requires"):
        load_monai_notebook_config()


def test_notebook_config_uses_overrides(monkeypatch):
    for env_name in REQUIRED_NOTEBOOK_CONFIG.values():
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.delenv("PIXELS_MONAI_MODEL_ID", raising=False)

    config = load_monai_notebook_config(
        overrides={
            "volume": "main.schema.volume",
            "input_dicom_subpath": "input/series",
            "output_subpath": "output/run",
            "experiment_name": "/Users/example/pixels-monai-flow",
            "model_id": "MONAI/test_bundle",
        }
    )

    assert config.volume_root == "/Volumes/main/schema/volume"
    assert config.input_dicom_dir == "/Volumes/main/schema/volume/input/series"
    assert config.output_dir == "/Volumes/main/schema/volume/output/run"
    assert config.model_id == "MONAI/test_bundle"
    assert config.experiment_name == "/Users/example/pixels-monai-flow"


def test_notebook_config_uses_task_parameters(monkeypatch):
    for env_name in REQUIRED_NOTEBOOK_CONFIG.values():
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.delenv("PIXELS_MONAI_MODEL_ID", raising=False)

    class Widgets:
        values = {
            "volume": "main.schema.volume",
            "input_dicom_subpath": "input/series",
            "output_subpath": "output/run",
            "experiment_name": "/Users/example/pixels-monai-flow",
            "model_id": "MONAI/task_bundle",
        }

        def get(self, name):
            return self.values[name]

    class Dbutils:
        widgets = Widgets()

    config = load_monai_notebook_config(dbutils=Dbutils())

    assert config.model_id == "MONAI/task_bundle"
    assert config.output_dir == "/Volumes/main/schema/volume/output/run"


def test_notebook_top_cell_has_blank_public_config():
    notebook = Path("09-MONAI-Flow-Bundle-Inference.py").read_text(encoding="utf-8")

    assert 'PIXELS_MONAI_VOLUME = ""' in notebook
    assert 'PIXELS_MONAI_INPUT_DICOM_SUBPATH = ""' in notebook
    assert 'PIXELS_MONAI_OUTPUT_SUBPATH = ""' in notebook
    assert 'PIXELS_MONAI_EXPERIMENT_NAME = ""' in notebook
    assert "catalog.schema.volume_name" not in notebook
    assert "edsp-wwfo-ses-itservices" not in notebook
    assert "dbutils.widgets.text" not in notebook


def test_relax_pipeline_generator_python_constraint(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('requires-python = ">=3.10,<3.11"\n', encoding="utf-8")

    assert _relax_pipeline_generator_python_constraint(tmp_path)
    assert pyproject.read_text(encoding="utf-8") == 'requires-python = ">=3.10"\n'


def test_transformer_stores_configuration():
    transformer = MonaiFlowBundleTransformer(
        modelUri="runs:/abc/model",
        inputCol="volume_path",
        outputCol="prediction",
        outputDir="/Volumes/main/schema/vol/output",
        labelPrompt=[3],
        modality="CT_BODY",
        numPartitions=2,
    )

    assert transformer.modelUri == "runs:/abc/model"
    assert transformer.inputCol == "volume_path"
    assert transformer.outputCol == "prediction"
    assert transformer.outputDir == "/Volumes/main/schema/vol/output"
    assert transformer.labelPrompt == [3]
    assert transformer.modality == "CT_BODY"
    assert transformer.numPartitions == 2
