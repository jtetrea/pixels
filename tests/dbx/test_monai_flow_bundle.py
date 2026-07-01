from pathlib import Path

import pandas as pd
import pytest

from dbx.pixels.modelserving.bundles.monai_flow import (
    DEFAULT_DICOM_METADATA_POLICY,
    DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS,
    DICOMMetadataPreparation,
    MonaiDeployAppModel,
    MonaiFlowBundleTransformer,
    _build_prediction_payload,
    _get_code_paths,
    _log_pyfunc_model,
    _merge_requirements,
    _normalise_prediction_result,
    _prepare_dicom_input_path,
    _relax_pipeline_generator_python_constraint,
    generate_monai_deploy_app,
    log_monai_deploy_app,
    log_monai_flow_bundle,
)
from dbx.pixels.modelserving.bundles.monai_runtime import (
    MONAI_DEPLOY_PACKAGES,
    REQUIRED_NOTEBOOK_CONFIG,
    TRITON_CLIENT_PACKAGES,
    load_monai_notebook_config,
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
        dicom_metadata_policy="deidentified_safe",
    )

    assert payload == {
        "image_path": "/Volumes/main/schema/vol/input.nii.gz",
        "output_dir": "/Volumes/main/schema/vol/output",
        "label_prompt": "3,4",
        "modality": "CT_BODY",
        "dicom_metadata_policy": "deidentified_safe",
    }


def _write_minimal_dicom(
    path: Path, *, include_patient_birth_date: bool = True
) -> None:
    pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientName = "Test^Patient"
    dataset.PatientID = "test-patient"
    if include_patient_birth_date:
        dataset.PatientBirthDate = ""
    dataset.PatientSex = ""
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.StudyID = ""
    dataset.StudyDate = ""
    dataset.StudyTime = ""
    dataset.AccessionNumber = ""
    dataset.Modality = "CT"
    dataset.Rows = 1
    dataset.Columns = 1
    dataset.save_as(str(path), enforce_file_format=True)


def test_prepare_dicom_input_repairs_missing_patient_birth_date(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    input_dir = tmp_path / "dicom"
    input_dir.mkdir()
    original = input_dir / "1.dcm"
    _write_minimal_dicom(original, include_patient_birth_date=False)

    prepared = _prepare_dicom_input_path(str(input_dir), policy="deidentified_safe")

    try:
        assert prepared.policy == DEFAULT_DICOM_METADATA_POLICY
        assert prepared.files_checked == 1
        assert prepared.files_repaired == 1
        assert "PatientBirthDate" in prepared.repaired_tags
        source = pydicom.dcmread(str(original), stop_before_pixels=True)
        repaired = pydicom.dcmread(
            str(Path(prepared.output_path) / "1.dcm"),
            stop_before_pixels=True,
        )
        assert not hasattr(source, "PatientBirthDate")
        assert repaired.PatientBirthDate == ""
    finally:
        if prepared.temp_dir:
            import shutil

            shutil.rmtree(prepared.temp_dir, ignore_errors=True)


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


def test_merge_requirements_keeps_first_package_specifier():
    requirements = _merge_requirements(
        ["monai>=1.5", "pydicom>=2.3"],
        ["monai==1.5.0", "highdicom"],
    )

    assert requirements == ["monai>=1.5", "pydicom>=2.3", "highdicom"]


def test_runtime_requirements_do_not_downgrade_databricks_ai_runtime_v5():
    assert "mlflow>=3,<4" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "mlflow>=2.17,<3.0" not in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert not any(
        req.startswith("databricks-connect")
        for req in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    )
    assert "monai-deploy-app-sdk==3.5.0" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "holoscan==3.10.0" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "holoscan-cu12==3.10.0" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "protobuf>=5.26.1,<6.0dev" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "grpcio>=1.67.1,<1.68" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "grpcio-status>=1.67.1,<1.68" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS
    assert "tritonclient[http,grpc]==2.60.0" in DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS


def test_notebook_runtime_packages_avoid_managed_databricks_runtime_pins():
    assert "monai==1.5.1" in MONAI_DEPLOY_PACKAGES
    assert "monai-deploy-app-sdk==3.5.0" in MONAI_DEPLOY_PACKAGES
    assert "holoscan-cu12==3.10.0" in MONAI_DEPLOY_PACKAGES
    assert "lazy-loader>=0.4" in MONAI_DEPLOY_PACKAGES
    assert "python-utils>=3.8" in MONAI_DEPLOY_PACKAGES
    assert "grpcio>=1.67.1,<1.68" in TRITON_CLIENT_PACKAGES
    assert "grpcio-status>=1.67.1,<1.68" in TRITON_CLIENT_PACKAGES
    assert "tritonclient[http,grpc]==2.60.0" in TRITON_CLIENT_PACKAGES
    assert not any(req.startswith("mlflow") for req in MONAI_DEPLOY_PACKAGES)
    assert not any(
        req.startswith("databricks-connect") for req in MONAI_DEPLOY_PACKAGES
    )


def test_deploy_app_model_normalizes_inputs():
    model = MonaiDeployAppModel()

    assert model._normalize_input("/Volumes/main/schema/vol/input") == {
        "image_path": "/Volumes/main/schema/vol/input"
    }
    assert model._normalize_input([{"image_path": "a", "output_dir": "b"}]) == {
        "image_path": "a",
        "output_dir": "b",
    }


def test_deploy_app_model_uses_repaired_dicom_input(monkeypatch, tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "app.py").write_text("print('app')\n", encoding="utf-8")
    (app_dir / "model").mkdir()
    input_dir = tmp_path / "dicom"
    input_dir.mkdir()
    repaired_dir = tmp_path / "repaired"
    repaired_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    temp_dir = tmp_path / "repair-temp"
    temp_dir.mkdir()
    captured = {}

    def fake_prepare(input_path, policy):
        captured["input_path"] = input_path
        captured["policy"] = policy
        return DICOMMetadataPreparation(
            input_path=input_path,
            output_path=str(repaired_dir),
            policy=policy,
            files_checked=1,
            files_repaired=1,
            repaired_tags=("PatientBirthDate",),
            temp_dir=str(temp_dir),
        )

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, cwd, env, capture_output, text, timeout):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        (output_dir / "seg.dcm").write_text("seg", encoding="utf-8")
        return Completed()

    monkeypatch.setattr(
        "dbx.pixels.modelserving.bundles.monai_flow._prepare_dicom_input_path",
        fake_prepare,
    )
    monkeypatch.setattr(
        "dbx.pixels.modelserving.bundles.monai_flow.subprocess.run",
        fake_run,
    )

    model = MonaiDeployAppModel()
    model.app_dir = str(app_dir)
    result = model.predict(
        None,
        {
            "image_path": str(input_dir),
            "output_dir": str(output_dir),
        },
    )

    assert captured["input_path"] == str(input_dir)
    assert captured["policy"] == DEFAULT_DICOM_METADATA_POLICY
    assert captured["command"][captured["command"].index("-i") + 1] == str(repaired_dir)
    assert captured["env"]["BUNDLE_PATH"] == str(app_dir / "model")
    assert not temp_dir.exists()
    assert result["output_files"] == [str(output_dir / "seg.dcm")]
    assert result["dicom_metadata_repairs"] == {
        "files_checked": 1,
        "files_repaired": 1,
        "repaired_tags": ["PatientBirthDate"],
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
            calls.append({"artifact_path": artifact_path, "python_model": python_model})

    class FakeMlflow:
        pyfunc = FakePyfunc()

    model = object()
    _log_pyfunc_model(FakeMlflow(), name="monai_deploy_app_model", python_model=model)

    assert calls == [{"artifact_path": "monai_deploy_app_model", "python_model": model}]


def test_code_paths_point_at_src_root():
    assert _get_code_paths() == [str(Path.cwd() / "src")]


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


def test_notebook_top_cell_has_blank_public_config():
    notebook = Path("notebooks/09-MONAI-Flow-Bundle-Inference.py").read_text(
        encoding="utf-8"
    )

    assert 'PIXELS_MONAI_VOLUME = ""' in notebook
    assert 'PIXELS_MONAI_INPUT_DICOM_SUBPATH = ""' in notebook
    assert 'PIXELS_MONAI_OUTPUT_SUBPATH = ""' in notebook
    assert 'PIXELS_MONAI_EXPERIMENT_NAME = ""' in notebook
    assert "catalog.schema.volume_name" not in notebook
    assert "edsp-wwfo-ses-itservices" not in notebook
    assert "dbutils.widgets.text" not in notebook
    assert "def _add_workspace_src_to_path()" in notebook
    assert 'Path("/Workspace" + notebook_path)' in notebook


def test_model_specific_monai_notebooks_select_expected_bundles():
    expected = {
        "notebooks/09a-MONAI-Flow-Spleen-DICOM.py": "MONAI/spleen_ct_segmentation",
        "notebooks/09b-MONAI-Flow-Pancreas-DICOM.py": (
            "MONAI/pancreas_ct_dints_segmentation"
        ),
        "notebooks/09c-MONAI-Flow-WholeBody-DICOM.py": (
            "MONAI/wholeBody_ct_segmentation"
        ),
    }

    for notebook_path, model_id in expected.items():
        notebook = Path(notebook_path).read_text(encoding="utf-8")

        assert 'PIXELS_MONAI_VOLUME = ""' in notebook
        assert 'PIXELS_MONAI_INPUT_DICOM_SUBPATH = ""' in notebook
        assert 'PIXELS_MONAI_OUTPUT_SUBPATH = ""' in notebook
        assert 'PIXELS_MONAI_EXPERIMENT_NAME = ""' in notebook
        assert f'PIXELS_MONAI_MODEL_ID = "{model_id}"' in notebook
        assert "catalog.schema.volume_name" not in notebook
        assert "dbutils.widgets.text" not in notebook


def test_relax_pipeline_generator_python_constraint(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('requires-python = ">=3.10,<3.11"\n', encoding="utf-8")

    assert _relax_pipeline_generator_python_constraint(tmp_path)
    assert pyproject.read_text(encoding="utf-8") == 'requires-python = ">=3.10"\n'


def test_generate_deploy_app_patches_segment_metadata(monkeypatch, tmp_path):
    app_dir = tmp_path / "pancreas_app"

    def fake_run_command(command, cwd=None):
        assert command[:2] == ["pg", "gen"]
        (app_dir / "model" / "configs").mkdir(parents=True)
        (app_dir / "model" / "configs" / "metadata.json").write_text(
            """
            {
              "network_data_format": {
                "outputs": {
                  "pred": {
                    "label_def": {
                      "0": "background",
                      "1": "pancreas",
                      "2": "pancreatic tumor"
                    }
                  }
                }
              }
            }
            """,
            encoding="utf-8",
        )
        (app_dir / "app.py").write_text(
            '''
from highdicom.sr.coding import codes
from monai.deploy.operators.dicom_seg_writer_operator import SegmentDescription

class App:
    def compose(self):
        segment_descriptions = [
            SegmentDescription(
                segment_label="Generated",
                segmented_property_category=codes.SCT.Organ,
                segmented_property_type=codes.SCT.Organ,
                algorithm_name="volumetric segmentation",
                algorithm_family=codes.DCM.ArtificialIntelligence,
                algorithm_version="0.1",
            )
        ]

        custom_tags = {}
''',
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "dbx.pixels.modelserving.bundles.monai_flow.ensure_monai_deploy_pipeline_generator",
        lambda **_kwargs: "pg",
    )
    monkeypatch.setattr(
        "dbx.pixels.modelserving.bundles.monai_flow._run_command",
        fake_run_command,
    )

    assert (
        generate_monai_deploy_app(
            model_id="MONAI/pancreas_ct_dints_segmentation",
            output_dir=str(app_dir),
        )
        == str(app_dir)
    )

    patched = (app_dir / "app.py").read_text(encoding="utf-8")
    assert 'segment_label="Pancreas"' in patched
    assert 'segment_label="Pancreatic Tumor"' in patched
    assert patched.count("SegmentDescription(") == 2
    assert "codes.SCT.Pancreas" in patched
    assert "codes.SCT.Neoplasm" in patched


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
