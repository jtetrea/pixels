import importlib

import pandas as pd
import pytest

from dbx.pixels.modelserving.bundles.monai_flow import (
    MissingMonaiFlowDependency,
    MonaiFlowBundleTransformer,
    _build_prediction_payload,
    _normalise_prediction_result,
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


def test_log_monai_flow_bundle_reports_missing_optional_dependency(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "monai_flow":
            raise ImportError("missing")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(MissingMonaiFlowDependency):
        log_monai_flow_bundle(bundle_name="spleen_ct_segmentation")


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
