from __future__ import annotations

import importlib
import logging
import os
import site
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

BOOTSTRAP_PACKAGES = [
    "filelock",
    "wheel-axle-runtime<1.0",
]
DATABRICKS_SERVERLESS_PACKAGES = [
    "mlflow>=2.17,<3.0",
    "databricks-connect>=15.4.2,<16",
]
MONAI_DEPLOY_PACKAGES = [
    "setuptools<82",
    "wheel",
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
    "scipy",
    "scikit-image",
    "lazy-loader>=0.4",
    "Pillow",
    "pytorch-ignite>=0.4",
    "numpy-stl>=3.0",
    "python-utils>=3.8",
    "trimesh",
]
TRITON_CLIENT_PACKAGES = [
    "protobuf>=5.26.1,<6.0dev",
    "grpcio>=1.67.1,<1.68",
    "grpcio-status>=1.67.1,<1.68",
    "tritonclient[http,grpc]==2.60.0",
]

REQUIRED_NOTEBOOK_CONFIG = {
    "volume": "PIXELS_MONAI_VOLUME",
    "input_dicom_subpath": "PIXELS_MONAI_INPUT_DICOM_SUBPATH",
    "output_subpath": "PIXELS_MONAI_OUTPUT_SUBPATH",
    "experiment_name": "PIXELS_MONAI_EXPERIMENT_NAME",
}


@dataclass(frozen=True)
class MonaiNotebookConfig:
    volume_name: str
    input_dicom_subpath: str
    output_subpath: str
    experiment_name: str
    model_id: str
    work_root: Path

    @property
    def volume_root(self) -> str:
        return "/Volumes/" + self.volume_name.replace(".", "/")

    @property
    def input_dicom_dir(self) -> str:
        return f"{self.volume_root}/{self.input_dicom_subpath.strip('/')}"

    @property
    def output_dir(self) -> str:
        return f"{self.volume_root}/{self.output_subpath.strip('/')}"


def quiet_known_runtime_warnings() -> None:
    """Reduce Databricks notebook noise while preserving real failures."""
    warnings.filterwarnings(
        "ignore",
        message=r".*Current stack size .* recommended minimum .*",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*pynvml package is deprecated.*",
        category=FutureWarning,
    )
    logging.getLogger("pyspark.sql.connect.logging").setLevel(logging.ERROR)
    os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    sys.dont_write_bytecode = True


def _pip_install(packages: Sequence[str], *, no_deps: bool) -> None:
    from pip._internal.cli.main import main as pip_main

    args = ["install", "-q", "--disable-pip-version-check"]
    if no_deps:
        args.append("--no-deps")
    args.extend(packages)
    exit_code = pip_main(args)
    if exit_code:
        raise RuntimeError(f"pip install failed with exit code {exit_code}: {list(packages)}")


def _version_tuple(version: str) -> tuple:
    parts = []
    for token in str(version).split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _version_in_range(version: str, min_version: str, max_version: str) -> bool:
    parsed = _version_tuple(version)
    return parsed >= _version_tuple(min_version) and parsed < _version_tuple(max_version)


def _prefer_distribution_site(
    distribution_name: str,
    min_version: str,
    max_version: str,
) -> tuple[Optional[str], Optional[str]]:
    import importlib.metadata as metadata

    roots = []
    for entry in list(sys.path) + site.getsitepackages() + [site.getusersitepackages()]:
        if not entry:
            continue
        root = Path(entry).resolve()
        if not root.is_dir() or root in roots:
            continue
        roots.append(root)

    candidates = []
    for root in roots:
        for dist in metadata.distributions(path=[str(root)]):
            name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
            if name == distribution_name.lower().replace("_", "-"):
                candidates.append((dist.version, str(root)))

    candidates.sort(key=lambda item: _version_tuple(item[0]), reverse=True)
    for version, root in candidates:
        if _version_in_range(version, min_version, max_version):
            sys.path[:] = [
                entry for entry in sys.path if str(Path(entry).resolve()) != root
            ]
            sys.path.insert(0, root)
            return version, root
    return None, None


def _ensure_imported_runtime(
    distribution_name: str,
    module_name: str,
    min_version: str,
    max_version: str,
) -> None:
    expected_version, expected_root = _prefer_distribution_site(
        distribution_name, min_version, max_version
    )
    if expected_version is None:
        raise RuntimeError(
            f"{distribution_name} {min_version} <= version < {max_version} was not installed."
        )

    loaded = sys.modules.get(module_name)
    loaded_version = getattr(loaded, "__version__", "") if loaded is not None else ""
    if loaded is not None and not _version_in_range(
        loaded_version, min_version, max_version
    ):
        for name in list(sys.modules):
            if name == module_name or name.startswith(module_name + "."):
                del sys.modules[name]

    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    active_version = getattr(module, "__version__", "")
    if not _version_in_range(active_version, min_version, max_version):
        raise RuntimeError(
            f"Active {module_name} is {active_version} from "
            f"{getattr(module, '__file__', 'unknown')}; expected "
            f"{distribution_name} {min_version} <= version < {max_version} "
            f"from {expected_root}."
        )


def _activate_holoscan_wheel_axle() -> list[str]:
    import wheel_axle.runtime

    finalized = []
    site_dirs = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        site_dirs.append(user_site)
    for site_dir in site_dirs:
        for pth_path in Path(site_dir).glob("holoscan_cu12-*.pth"):
            wheel_axle.runtime.finalize(str(pth_path))
            finalized.append(str(pth_path))
    return finalized


def ensure_monai_runtime() -> None:
    """Install and activate the MONAI Deploy stack used by Notebook 9."""
    quiet_known_runtime_warnings()
    _pip_install(BOOTSTRAP_PACKAGES, no_deps=True)
    _pip_install(MONAI_DEPLOY_PACKAGES, no_deps=True)
    _pip_install(DATABRICKS_SERVERLESS_PACKAGES + TRITON_CLIENT_PACKAGES, no_deps=False)
    importlib.invalidate_caches()
    _ensure_imported_runtime("grpcio", "grpc", "1.67.1", "1.68")
    try:
        _activate_holoscan_wheel_axle()
    except Exception as exc:
        raise RuntimeError("Failed to activate holoscan-cu12 wheel-axle runtime") from exc
    importlib.invalidate_caches()


def _task_parameter(dbutils: Any, name: str) -> Optional[str]:
    if dbutils is None:
        return None
    try:
        value = dbutils.widgets.get(name)
    except Exception:
        return None
    value = str(value).strip()
    return value or None


def _config_value(
    name: str,
    env_name: str,
    dbutils: Any,
    overrides: Mapping[str, str],
) -> Optional[str]:
    value = overrides.get(name)
    if value and value.strip():
        return value.strip()
    value = os.environ.get(env_name)
    if value and value.strip():
        return value.strip()
    return _task_parameter(dbutils, name)


def load_monai_notebook_config(
    *,
    dbutils: Any = None,
    overrides: Optional[Mapping[str, str]] = None,
    model_id: str = "MONAI/spleen_ct_segmentation",
) -> MonaiNotebookConfig:
    """Load Notebook 9 configuration from top-cell values, env vars, or task params."""
    overrides = overrides or {}
    values: Dict[str, str] = {}
    missing = []
    for name, env_name in REQUIRED_NOTEBOOK_CONFIG.items():
        value = _config_value(name, env_name, dbutils, overrides)
        if not value:
            missing.append(f"{name} ({env_name})")
            continue
        values[name] = value
        os.environ[env_name] = value

    if missing:
        raise ValueError(
            "Notebook 9 requires these configuration values: "
            + ", ".join(missing)
            + ". Set them in the first cell, environment variables, or Databricks task parameters."
        )

    selected_model_id = (
        (overrides.get("model_id") or "").strip()
        or os.environ.get("PIXELS_MONAI_MODEL_ID", "").strip()
        or _task_parameter(dbutils, "model_id")
        or model_id
    )
    os.environ["PIXELS_MONAI_MODEL_ID"] = selected_model_id
    return MonaiNotebookConfig(
        volume_name=values["volume"],
        input_dicom_subpath=values["input_dicom_subpath"],
        output_subpath=values["output_subpath"],
        experiment_name=values["experiment_name"],
        model_id=selected_model_id,
        work_root=Path(tempfile.mkdtemp(prefix="pixels_monai_flow_")),
    )


def validate_monai_runtime() -> Dict[str, str]:
    """Import key runtime packages and return concise version details."""
    quiet_known_runtime_warnings()

    import importlib.metadata as metadata

    import holoscan
    import mlflow
    import monai
    import monai.deploy.conditions  # noqa: F401
    import torch

    details = {
        "monai": monai.__version__,
        "monai-deploy": metadata.version("monai-deploy-app-sdk"),
        "holoscan-cu12": metadata.version("holoscan-cu12"),
        "holoscan": holoscan.__version__,
        "torch": torch.__version__,
        "mlflow": mlflow.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        details["gpu"] = torch.cuda.get_device_name(0)
    return details


def print_runtime_summary(details: Mapping[str, str]) -> None:
    versions = ", ".join(
        f"{name} {details[name]}"
        for name in ["monai", "monai-deploy", "holoscan-cu12", "holoscan", "torch", "mlflow"]
        if name in details
    )
    print(versions)
    print(f"CUDA available: {details.get('cuda_available', 'unknown')}")
    if details.get("gpu"):
        print(f"GPU: {details['gpu']}")
