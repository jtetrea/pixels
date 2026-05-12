import pandas as pd
import pytest

from dbx.pixels.modelserving.bundles.monai_flow import (
    MonaiDeployAppModel,
    MonaiFlowBundleTransformer,
    _build_prediction_payload,
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


def test_runtime_requirements_pin_non_breaking_holoscan():
    from dbx.pixels.modelserving.bundles.monai_flow import (
        DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS,
    )

    assert "monai-deploy-app-sdk==3.5.0" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "holoscan-cu12==4.0.0" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS


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
