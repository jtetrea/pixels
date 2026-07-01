from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from pyspark.ml.pipeline import Transformer
except (
    ImportError
):  # pragma: no cover - local unit-test fallback when Spark is unavailable

    class Transformer:  # type: ignore[no-redef]
        def transform(self, df):
            return self._transform(df)


try:
    import mlflow.pyfunc

    _PythonModelBase = mlflow.pyfunc.PythonModel
except ImportError:  # pragma: no cover - allows local helper tests without MLflow
    mlflow = None  # type: ignore[assignment]

    class _PythonModelBase:  # type: ignore[no-redef]
        pass


DEFAULT_PIPELINE_GENERATOR_REPO = (
    "https://github.com/Project-MONAI/monai-deploy-app-sdk.git"
)
DEFAULT_PIPELINE_GENERATOR_BRANCH = "main"
DEFAULT_PIPELINE_GENERATOR_DIR = "/tmp/pixels_monai_flow/tools/pipeline-generator"

STRICT_DICOM_METADATA_POLICY = "strict"
DEIDENTIFIED_SAFE_DICOM_METADATA_POLICY = "deidentified_safe"
DEFAULT_DICOM_METADATA_POLICY = DEIDENTIFIED_SAFE_DICOM_METADATA_POLICY
SUPPORTED_DICOM_METADATA_POLICIES = {
    STRICT_DICOM_METADATA_POLICY,
    DEIDENTIFIED_SAFE_DICOM_METADATA_POLICY,
}


@dataclass(frozen=True)
class SegmentSpec:
    label: str
    category_expr: str
    type_expr: str


_SEGMENT_CODE_MAP: Dict[str, Tuple[str, str]] = {
    "artery": ("codes.SCT.BloodVessel", "codes.SCT.Artery"),
    "gallbladder": ("codes.SCT.Organ", "codes.SCT.Gallbladder"),
    "liver": ("codes.SCT.Organ", "codes.SCT.Liver"),
    "pancreas": ("codes.SCT.Organ", "codes.SCT.Pancreas"),
    "pancreatic tumor": (
        "codes.SCT.MorphologicallyAbnormalStructure",
        "codes.SCT.Neoplasm",
    ),
    "portal vein": ("codes.SCT.BloodVessel", "codes.SCT.PortalVein"),
    "spleen": ("codes.SCT.Organ", "codes.SCT.Spleen"),
    "stomach": ("codes.SCT.Organ", "codes.SCT.Stomach"),
    "tumor": ("codes.SCT.MorphologicallyAbnormalStructure", "codes.SCT.Neoplasm"),
}

# highdicom.seg.Segmentation reads these source-image attributes directly.
# Deidentified data may remove Type 2 tags entirely; zero-length values preserve
# redaction while satisfying pydicom attribute access.
DEIDENTIFIED_SAFE_EMPTY_TAGS: Tuple[str, ...] = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "AccessionNumber",
    "StudyID",
    "StudyDate",
    "StudyTime",
)

DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS = [
    "setuptools<82",
    "wheel",
    "filelock",
    "wheel-axle-runtime<1.0",
    "mlflow>=3,<4",
    "monai==1.5.1",
    "monai-deploy-app-sdk==3.5.0",
    "holoscan==3.10.0",
    "holoscan-cu12==3.10.0",
    "colorama>=0.4.1",
    "typeguard>=3.0.0",
    "pydicom>=2.3",
    "highdicom",
    "pyjpegls",
    "nibabel",
    "SimpleITK>=2.0",
    "numpy",
    "scipy",
    "scikit-image",
    "lazy-loader>=0.4",
    "Pillow",
    "numpy-stl>=3.0",
    "python-utils>=3.8",
    "trimesh",
    "protobuf>=5.26.1,<6.0dev",
    "grpcio>=1.67.1,<1.68",
    "grpcio-status>=1.67.1,<1.68",
    "tritonclient[http,grpc]==2.60.0",
    "torch",
    "torchvision",
    "pytorch-ignite>=0.4",
]


class MissingMonaiRuntimeDependency(ImportError):
    """Raised when optional MONAI runtime dependencies are not installed."""


@dataclass(frozen=True)
class DICOMMetadataPreparation:
    input_path: str
    output_path: str
    policy: str
    files_checked: int = 0
    files_repaired: int = 0
    repaired_tags: Tuple[str, ...] = ()
    temp_dir: Optional[str] = None


def _normalize_dicom_metadata_policy(policy: Optional[str]) -> str:
    resolved = (policy or DEFAULT_DICOM_METADATA_POLICY).strip().lower()
    if resolved not in SUPPORTED_DICOM_METADATA_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_DICOM_METADATA_POLICIES))
        raise ValueError(
            f"Unsupported dicom_metadata_policy: {policy}. Supported: {supported}."
        )
    return resolved


def _missing_deidentified_safe_tags(dataset: object) -> Tuple[str, ...]:
    return tuple(
        tag for tag in DEIDENTIFIED_SAFE_EMPTY_TAGS if not hasattr(dataset, tag)
    )


def _repair_deidentified_safe_dataset(dataset: object) -> Tuple[str, ...]:
    missing = _missing_deidentified_safe_tags(dataset)
    for tag in missing:
        setattr(dataset, tag, "")
    return missing


def _iter_candidate_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for child in sorted(path.rglob("*")):
        if child.is_file():
            yield child


def _read_dicom_headers(path: Path):
    import pydicom

    for candidate in _iter_candidate_files(path):
        try:
            yield candidate, pydicom.dcmread(str(candidate), stop_before_pixels=True)
        except Exception:
            continue


def _copy_one_with_deidentified_repairs(source: Path, destination: Path) -> int:
    import pydicom

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataset = pydicom.dcmread(str(source))
    except Exception:
        shutil.copy2(source, destination)
        return 0

    if _repair_deidentified_safe_dataset(dataset):
        dataset.save_as(str(destination))
        return 1

    shutil.copy2(source, destination)
    return 0


def _copy_with_deidentified_repairs(source: Path, destination: Path) -> int:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        return _copy_one_with_deidentified_repairs(source, destination)

    repaired = 0
    destination.mkdir(parents=True, exist_ok=True)
    for candidate in _iter_candidate_files(source):
        repaired += _copy_one_with_deidentified_repairs(
            candidate,
            destination / candidate.relative_to(source),
        )
    return repaired


def _prepare_dicom_input_path(
    input_path: str,
    policy: Optional[str] = None,
) -> DICOMMetadataPreparation:
    resolved_policy = _normalize_dicom_metadata_policy(policy)
    input_path = os.path.abspath(input_path)
    if resolved_policy == STRICT_DICOM_METADATA_POLICY:
        return DICOMMetadataPreparation(
            input_path=input_path,
            output_path=input_path,
            policy=resolved_policy,
        )

    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    dicom_files = []
    repaired_tags = set()
    for _path, dataset in _read_dicom_headers(source):
        missing = _missing_deidentified_safe_tags(dataset)
        dicom_files.append(missing)
        repaired_tags.update(missing)

    if not repaired_tags:
        return DICOMMetadataPreparation(
            input_path=input_path,
            output_path=input_path,
            policy=resolved_policy,
            files_checked=len(dicom_files),
        )

    temp_dir = tempfile.mkdtemp(prefix="pixels_monai_dicom_metadata_")
    temp_root = Path(temp_dir) / source.name
    files_repaired = _copy_with_deidentified_repairs(source, temp_root)
    return DICOMMetadataPreparation(
        input_path=input_path,
        output_path=str(temp_root),
        policy=resolved_policy,
        files_checked=len(dicom_files),
        files_repaired=files_repaired,
        repaired_tags=tuple(sorted(repaired_tags)),
        temp_dir=temp_dir,
    )


def _require_mlflow():
    try:
        import mlflow as mlflow_module
        import mlflow.pyfunc  # noqa: F401
        from mlflow.models import infer_signature
    except ImportError as e:
        raise MissingMonaiRuntimeDependency(
            "MLflow is required for MONAI deploy app logging. Use a Databricks "
            "ML runtime or install mlflow before calling log_monai_flow_bundle."
        ) from e
    return mlflow_module, infer_signature


def _log_pyfunc_model(mlflow_module: Any, *, name: str, **kwargs: Any) -> None:
    """Call MLflow pyfunc logging with MLflow 2/3 compatible path naming."""
    if "name" in inspect.signature(mlflow_module.pyfunc.log_model).parameters:
        mlflow_module.pyfunc.log_model(name=name, **kwargs)
    else:
        mlflow_module.pyfunc.log_model(artifact_path=name, **kwargs)


def _run_command(command: Sequence[str], cwd: Optional[str] = None) -> None:
    subprocess.check_call(list(command), cwd=cwd)


def _pipeline_generator_executable() -> Optional[str]:
    pg = shutil.which("pg")
    if pg:
        return pg
    candidate = Path(sys.executable).with_name("pg")
    return str(candidate) if candidate.exists() else None


def _relax_pipeline_generator_python_constraint(target_dir: Path) -> bool:
    pyproject = target_dir / "pyproject.toml"
    if not pyproject.exists():
        return False

    content = pyproject.read_text(encoding="utf-8")
    patched = content.replace(
        'requires-python = ">=3.10,<3.11"',
        'requires-python = ">=3.10"',
    )
    if patched == content:
        return False
    pyproject.write_text(patched, encoding="utf-8")
    return True


def _normalize_segment_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.replace("_", " ").strip().lower())


def _segment_spec(label: str) -> SegmentSpec:
    category_expr, type_expr = _SEGMENT_CODE_MAP.get(
        _normalize_segment_label(label),
        ("codes.SCT.AnatomicalStructure", "codes.SCT.AnatomicalStructure"),
    )
    return SegmentSpec(
        label=label.title(),
        category_expr=category_expr,
        type_expr=type_expr,
    )


def _labels_from_bundle_metadata(metadata_path: Path) -> List[str]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    outputs = metadata.get("network_data_format", {}).get("outputs", {})
    labels: Dict[int, str] = {}

    for output in outputs.values():
        label_def = output.get("label_def") or output.get("channel_def") or {}
        for raw_index, raw_label in label_def.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            label = str(raw_label).strip()
            if index == 0 or _normalize_segment_label(label) == "background":
                continue
            labels[index] = label

    return [labels[index] for index in sorted(labels)]


def _extract_algorithm_metadata(app_text: str) -> Tuple[str, str]:
    name_match = re.search(r'algorithm_name="([^"]+)"', app_text)
    version_match = re.search(r'algorithm_version="([^"]+)"', app_text)
    return (
        name_match.group(1)
        if name_match
        else "volumetric medical image segmentation",
        version_match.group(1) if version_match else "unknown",
    )


def _render_segment_descriptions(
    labels: Iterable[str],
    algorithm_name: str,
    algorithm_version: str,
) -> str:
    entries = []
    for label in labels:
        spec = _segment_spec(label)
        entries.append(
            "\n".join(
                [
                    "            SegmentDescription(",
                    f'                segment_label="{spec.label}",',
                    f"                segmented_property_category={spec.category_expr},",
                    f"                segmented_property_type={spec.type_expr},",
                    f'                algorithm_name="{algorithm_name}",',
                    "                algorithm_family=codes.DCM.ArtificialIntelligence,",
                    f'                algorithm_version="{algorithm_version}",',
                    "            )",
                ]
            )
        )

    return "        segment_descriptions = [\n" + ",\n".join(entries) + "\n        ]"


def _patch_deploy_app_segments(app_dir: Path) -> Optional[List[str]]:
    app_py = app_dir / "app.py"
    metadata_path = app_dir / "model" / "configs" / "metadata.json"
    if not app_py.is_file() or not metadata_path.is_file():
        return None

    labels = _labels_from_bundle_metadata(metadata_path)
    if not labels:
        return None

    app_text = app_py.read_text(encoding="utf-8")
    algorithm_name, algorithm_version = _extract_algorithm_metadata(app_text)
    segment_block = _render_segment_descriptions(
        labels,
        algorithm_name,
        algorithm_version,
    )
    pattern = re.compile(
        r"        segment_descriptions = \[\n.*?\n        \]\n\n        custom_tags",
        re.DOTALL,
    )
    patched, count = pattern.subn(
        f"{segment_block}\n\n        custom_tags",
        app_text,
        count=1,
    )
    if count == 0:
        return None

    app_py.write_text(patched, encoding="utf-8")
    return labels


def _target_name(transform: Dict[str, Any]) -> str:
    return str(transform.get("_target_", "")).rsplit(".", 1)[-1]


def _has_argmax(transform: Dict[str, Any]) -> bool:
    argmax = transform.get("argmax")
    if isinstance(argmax, list):
        return any(bool(value) for value in argmax)
    return bool(argmax)


def _disable_softmax_before_argmax(value: Any) -> int:
    changes = 0
    if isinstance(value, dict):
        for child in value.values():
            changes += _disable_softmax_before_argmax(child)
        return changes

    if not isinstance(value, list):
        return 0

    for index, item in enumerate(value):
        changes += _disable_softmax_before_argmax(item)
        if not isinstance(item, dict):
            continue
        if _target_name(item) != "Activationsd" or item.get("softmax") is not True:
            continue
        has_following_argmax = any(
            isinstance(candidate, dict)
            and _target_name(candidate) == "AsDiscreted"
            and _has_argmax(candidate)
            for candidate in value[index + 1 :]
        )
        if has_following_argmax:
            item["softmax"] = False
            changes += 1
    return changes


def _patch_large_multiclass_softmax(app_dir: Path, label_count: int) -> List[str]:
    if label_count < 16:
        return []

    config_dir = app_dir / "model" / "configs"
    if not config_dir.is_dir():
        return []

    changes = []
    for config_path in sorted(config_dir.glob("*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        count = _disable_softmax_before_argmax(config)
        if count:
            config_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
            changes.append(f"{config_path.name}: disabled {count} softmax transform(s)")
    return changes


def _patch_generated_deploy_app(app_dir: Path) -> None:
    labels = _patch_deploy_app_segments(app_dir) or []
    _patch_large_multiclass_softmax(app_dir, label_count=len(labels))


def ensure_monai_deploy_pipeline_generator(
    install_dir: str = DEFAULT_PIPELINE_GENERATOR_DIR,
    repo_url: str = DEFAULT_PIPELINE_GENERATOR_REPO,
    branch: str = DEFAULT_PIPELINE_GENERATOR_BRANCH,
) -> str:
    """Install the MONAI Deploy pipeline-generator CLI when it is unavailable."""

    existing_pg = _pipeline_generator_executable()
    if existing_pg:
        return existing_pg

    target_dir = Path(install_dir)
    if not (target_dir / "pipeline_generator").is_dir():
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="pixels_monai_deploy_") as tmp_dir:
            clone_dir = Path(tmp_dir) / "repo"
            _run_command(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    branch,
                    repo_url,
                    str(clone_dir),
                ]
            )
            source_dir = clone_dir / "tools" / "pipeline-generator"
            if not source_dir.is_dir():
                raise FileNotFoundError(
                    f"pipeline-generator not found in cloned MONAI repo: {source_dir}"
                )
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(source_dir, target_dir)

    _relax_pipeline_generator_python_constraint(target_dir)
    _run_command([sys.executable, "-m", "pip", "install", "-q", "-e", str(target_dir)])
    pg = _pipeline_generator_executable()
    if not pg:
        raise FileNotFoundError(
            "Installed MONAI pipeline-generator, but could not find the `pg` "
            "executable in PATH or next to the current Python executable."
        )
    return pg


def generate_monai_deploy_app(
    model_id: str,
    output_dir: str,
    app_format: str = "dicom",
    force: bool = True,
    pipeline_generator_dir: str = DEFAULT_PIPELINE_GENERATOR_DIR,
    repo_url: str = DEFAULT_PIPELINE_GENERATOR_REPO,
    branch: str = DEFAULT_PIPELINE_GENERATOR_BRANCH,
) -> str:
    """Download a public MONAI bundle and generate a MONAI Deploy app."""

    if not model_id:
        raise ValueError("model_id is required.")
    deploy_app_dir = Path(output_dir)
    if deploy_app_dir.exists() and force:
        shutil.rmtree(deploy_app_dir)
    deploy_app_dir.parent.mkdir(parents=True, exist_ok=True)

    pg = ensure_monai_deploy_pipeline_generator(
        install_dir=pipeline_generator_dir,
        repo_url=repo_url,
        branch=branch,
    )
    command = [
        pg,
        "gen",
        model_id,
        "--format",
        app_format,
        "--output",
        str(deploy_app_dir),
    ]
    if force:
        command.append("-f")
    _run_command(command)

    app_py = deploy_app_dir / "app.py"
    if not app_py.exists():
        raise FileNotFoundError(f"Generated app.py not found in {deploy_app_dir}")
    _patch_generated_deploy_app(deploy_app_dir)
    return str(deploy_app_dir)


def _load_requirements(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _merge_requirements(*groups: Iterable[str]) -> List[str]:
    seen = set()
    merged = []
    for group in groups:
        for requirement in group:
            requirement = requirement.strip()
            if not requirement or requirement.startswith("#"):
                continue
            key = requirement.split("==")[0].split(">=")[0].split("<")[0].lower()
            if key not in seen:
                seen.add(key)
                merged.append(requirement)
    return merged


def _copy_deploy_app_excluding_venv(source_dir: str) -> Tuple[str, str]:
    base_dir = tempfile.mkdtemp(prefix="pixels_monai_deploy_app_")
    artifact_dir = os.path.join(base_dir, "deploy_app")

    def ignore(_directory: str, names: List[str]) -> List[str]:
        return [name for name in names if name == ".venv"]

    shutil.copytree(source_dir, artifact_dir, ignore=ignore, symlinks=False)
    return artifact_dir, base_dir


def _get_code_paths() -> List[str]:
    return [str(Path(__file__).resolve().parents[4])]


class MonaiDeployAppModel(_PythonModelBase):
    """MLflow pyfunc wrapper for a generated MONAI Deploy application."""

    def __init__(self) -> None:
        self.app_dir: Optional[str] = None

    def load_context(self, context) -> None:
        self.app_dir = context.artifacts.get("deploy_app_dir")
        if not self.app_dir:
            raise ValueError("Missing deploy_app_dir artifact.")
        self.app_dir = os.path.abspath(self.app_dir)
        app_py = os.path.join(self.app_dir, "app.py")
        if not os.path.isfile(app_py):
            raise FileNotFoundError(
                f"app.py not found in deploy app directory: {self.app_dir}"
            )

    def _normalize_input(self, model_input: Any) -> Dict[str, Any]:
        if isinstance(model_input, dict):
            return model_input
        if hasattr(model_input, "to_dict") and callable(
            getattr(model_input, "to_dict")
        ):
            return model_input.to_dict(orient="records")[0] if len(model_input) else {}
        if isinstance(model_input, list) and model_input:
            if isinstance(model_input[0], dict):
                return model_input[0]
            return {"image_path": model_input[0]}
        if isinstance(model_input, str):
            return {"image_path": model_input}
        return {}

    def predict(self, context, model_input):
        if not self.app_dir:
            raise RuntimeError("Deploy app context not loaded.")

        payload = self._normalize_input(model_input)
        image_path = payload.get("image_path") or payload.get("image")
        if not image_path:
            raise ValueError("Missing image_path in model input.")
        image_path = os.path.abspath(str(image_path))
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input path not found: {image_path}")

        user_output_dir = payload.get("output_dir")
        user_output_dir = os.path.abspath(str(user_output_dir or "monai_output"))
        Path(user_output_dir).mkdir(parents=True, exist_ok=True)

        dicom_metadata_policy = (
            payload.get("dicom_metadata_policy")
            or os.environ.get("PIXELS_MONAI_DICOM_METADATA_POLICY")
            or DEFAULT_DICOM_METADATA_POLICY
        )
        prepared_input = _prepare_dicom_input_path(
            image_path,
            policy=str(dicom_metadata_policy),
        )

        python_exe = os.path.join(self.app_dir, ".venv", "bin", "python")
        if not os.path.isfile(python_exe):
            python_exe = sys.executable

        app_py = os.path.join(self.app_dir, "app.py")
        bundle_path = os.path.join(self.app_dir, "model")
        env = os.environ.copy()
        env["BUNDLE_PATH"] = bundle_path
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        try:
            result = subprocess.run(
                [
                    python_exe,
                    app_py,
                    "-i",
                    prepared_input.output_path,
                    "-o",
                    user_output_dir,
                ],
                cwd=self.app_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"MONAI Deploy app failed (exit {result.returncode}): "
                    f"{result.stderr or result.stdout}"
                )
        finally:
            if prepared_input.temp_dir:
                shutil.rmtree(prepared_input.temp_dir, ignore_errors=True)

        output_files = [
            str(path) for path in Path(user_output_dir).rglob("*") if path.is_file()
        ]
        return {
            "output_dir": user_output_dir,
            "output_files": output_files,
            "dicom_metadata_policy": prepared_input.policy,
            "dicom_metadata_repairs": {
                "files_checked": prepared_input.files_checked,
                "files_repaired": prepared_input.files_repaired,
                "repaired_tags": list(prepared_input.repaired_tags),
            },
        }


def log_monai_flow_bundle(
    deploy_app_dir: str,
    model_name: str = "monai_deploy_app_model",
    input_example_path: Optional[str] = None,
    requirements_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
    register_model: bool = False,
    registered_model_name: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[str, str]:
    """Log a generated MONAI Deploy app as an MLflow pyfunc model."""

    deploy_app_dir = os.path.abspath(deploy_app_dir)
    if not os.path.isdir(deploy_app_dir):
        raise FileNotFoundError(f"Deploy app directory not found: {deploy_app_dir}")
    app_py = os.path.join(deploy_app_dir, "app.py")
    if not os.path.isfile(app_py):
        raise FileNotFoundError(
            f"app.py not found in deploy app directory: {deploy_app_dir}"
        )

    mlflow_module, infer_signature = _require_mlflow()

    input_example = None
    if input_example_path:
        resolved_input_path = os.path.abspath(input_example_path)
        if not os.path.exists(resolved_input_path):
            raise FileNotFoundError(f"Input example not found: {resolved_input_path}")
        input_example = {
            "image_path": resolved_input_path,
            "output_dir": "/tmp/pixels_monai_flow_output",
        }
    else:
        resolved_input_path = "/tmp/pixels_monai_flow_input"

    output_example = {
        "output_dir": "/tmp/pixels_monai_flow_output",
        "output_files": ["/tmp/pixels_monai_flow_output/output.dcm"],
        "dicom_metadata_policy": DEFAULT_DICOM_METADATA_POLICY,
        "dicom_metadata_repairs": {
            "files_checked": 0,
            "files_repaired": 0,
            "repaired_tags": [],
        },
    }
    signature = infer_signature(
        model_input=input_example
        or {
            "image_path": resolved_input_path,
            "output_dir": "/tmp/pixels_monai_flow_output",
        },
        model_output=output_example,
    )

    app_requirements = _load_requirements(
        requirements_path or os.path.join(deploy_app_dir, "requirements.txt")
    )
    pip_requirements = _merge_requirements(
        DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS, app_requirements
    )

    if experiment_name:
        mlflow_module.set_experiment(experiment_name)

    artifact_dir, temp_base = _copy_deploy_app_excluding_venv(deploy_app_dir)
    try:
        with mlflow_module.start_run() as run:
            if verbose:
                print(f"Logging MONAI Deploy app to MLflow: {deploy_app_dir}")
                print(f"  Model name: {model_name}")
                if input_example_path:
                    print(f"  Input example: {resolved_input_path}")
            _log_pyfunc_model(
                mlflow_module,
                name=model_name,
                python_model=MonaiDeployAppModel(),
                artifacts={"deploy_app_dir": artifact_dir},
                pip_requirements=pip_requirements,
                signature=signature,
                input_example=input_example,
                code_paths=_get_code_paths(),
            )
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/{model_name}"
            if register_model:
                mlflow_module.register_model(
                    model_uri, registered_model_name or model_name
                )
            if verbose:
                print(f"  Run ID: {run_id}")
                print(f"  Model URI: {model_uri}")
            return run_id, model_uri
    finally:
        shutil.rmtree(temp_base, ignore_errors=True)


log_monai_deploy_app = log_monai_flow_bundle


def _coerce_label_prompt(label_prompt: Any) -> Optional[str]:
    if label_prompt is None:
        return None
    if isinstance(label_prompt, str):
        return label_prompt
    if isinstance(label_prompt, Iterable):
        return ",".join(str(v) for v in label_prompt)
    return str(label_prompt)


def _build_prediction_payload(
    image_path: str,
    output_dir: Optional[str] = None,
    label_prompt: Any = None,
    modality: Optional[str] = None,
    dicom_metadata_policy: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"image_path": image_path}
    if output_dir:
        payload["output_dir"] = output_dir
    prompt = _coerce_label_prompt(label_prompt)
    if prompt:
        payload["label_prompt"] = prompt
    if modality:
        payload["modality"] = modality
    if dicom_metadata_policy:
        payload["dicom_metadata_policy"] = _normalize_dicom_metadata_policy(
            dicom_metadata_policy
        )
    return payload


def _normalise_prediction_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, pd.DataFrame):
        if result.empty:
            result = {}
        else:
            result = result.iloc[0].to_dict()
    if not isinstance(result, dict):
        result = {"result": result}

    output_files = result.get("output_files", [])
    if output_files is None:
        output_files = []
    elif isinstance(output_files, str):
        output_files = [output_files]
    else:
        output_files = [str(path) for path in output_files]

    labels = result.get("labels", {}) or {}
    labels = {str(k): str(v) for k, v in dict(labels).items()}

    return {
        "output_dir": result.get("output_dir"),
        "output_files": output_files,
        "labels": labels,
        "error": result.get("error", ""),
    }


class MonaiFlowBundleTransformer(Transformer):
    """Run a logged MONAI Deploy MLflow pyfunc model against image paths."""

    def __init__(
        self,
        modelUri: str,
        inputCol: str = "image_path",
        outputCol: str = "monai_result",
        outputDir: Optional[str] = None,
        labelPrompt: Any = None,
        modality: Optional[str] = None,
        numPartitions: Optional[int] = None,
    ):
        self.modelUri = modelUri
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.outputDir = outputDir
        self.labelPrompt = labelPrompt
        self.modality = modality
        self.numPartitions = numPartitions

    def _validate_schema(self, df) -> None:
        import pyspark.sql.types as t

        field = df.schema[self.inputCol]
        if field.dataType != t.StringType():
            raise TypeError(
                f"MonaiFlowBundleTransformer field {self.inputCol}, input type "
                f"{field.dataType} did not match input type StringType"
            )

    def _transform(self, df):
        import pyspark.sql.functions as F
        import pyspark.sql.types as t

        self._validate_schema(df)

        model_uri = self.modelUri
        output_dir = self.outputDir
        label_prompt = self.labelPrompt
        modality = self.modality

        result_schema = t.StructType(
            [
                t.StructField("output_dir", t.StringType(), True),
                t.StructField("output_files", t.ArrayType(t.StringType()), True),
                t.StructField(
                    "labels", t.MapType(t.StringType(), t.StringType()), True
                ),
                t.StructField("error", t.StringType(), True),
            ]
        )

        @F.pandas_udf(result_schema)
        def predict_monai(paths: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
            import mlflow.pyfunc

            model = mlflow.pyfunc.load_model(model_uri)
            for batch in paths:
                rows = []
                for image_path in batch:
                    payload = _build_prediction_payload(
                        image_path=str(image_path),
                        output_dir=output_dir,
                        label_prompt=label_prompt,
                        modality=modality,
                    )
                    try:
                        rows.append(
                            _normalise_prediction_result(model.predict(payload))
                        )
                    except Exception as e:
                        rows.append(
                            {
                                "output_dir": output_dir,
                                "output_files": [],
                                "labels": {},
                                "error": str(e),
                            }
                        )
                yield pd.DataFrame(rows)

        if self.numPartitions is not None:
            df = df.repartition(self.numPartitions)
        return df.withColumn(self.outputCol, predict_monai(F.col(self.inputCol)))
