from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from pyspark.ml.pipeline import Transformer
except ImportError:  # pragma: no cover - local unit-test fallback when Spark is unavailable

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


DEFAULT_PIPELINE_GENERATOR_REPO = "https://github.com/Project-MONAI/monai-deploy-app-sdk.git"
DEFAULT_PIPELINE_GENERATOR_BRANCH = "main"
DEFAULT_PIPELINE_GENERATOR_DIR = "/tmp/pixels_monai_flow/tools/pipeline-generator"
DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS = [
    "setuptools<82",
    "monai>=1.5",
    "monai-deploy-app-sdk>=3.0",
    "pydicom>=2.3",
    "highdicom",
    "nibabel",
    "SimpleITK>=2.0",
    "numpy",
    "scipy",
    "scikit-image",
    "Pillow",
    "torch",
    "torchvision",
    "pytorch-ignite>=0.4",
]


class MissingMonaiRuntimeDependency(ImportError):
    """Raised when optional MONAI runtime dependencies are not installed."""


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


def _run_command(command: Sequence[str], cwd: Optional[str] = None) -> None:
    subprocess.check_call(list(command), cwd=cwd)


def _pipeline_generator_executable() -> Optional[str]:
    pg = shutil.which("pg")
    if pg:
        return pg

    candidate = Path(sys.executable).with_name("pg")
    if candidate.exists():
        return str(candidate)

    return None


def patch_monai_deploy_holoscan_compatibility(monai_package_dir: Optional[str] = None) -> bool:
    """Patch MONAI Deploy imports for Holoscan releases that renamed graphs.

    GPU serverless environments may pair monai-deploy-app-sdk 3.5 with
    holoscan 4.x. In that combination, MONAI Deploy imports holoscan.graphs,
    while Holoscan exposes the module as holoscan.flow_graphs.
    """

    if monai_package_dir is None:
        try:
            import monai
        except ImportError as e:
            raise MissingMonaiRuntimeDependency(
                "MONAI is required before applying the Holoscan compatibility patch."
            ) from e
        monai_package_dir = os.path.dirname(monai.__file__)

    graphs_init = Path(monai_package_dir) / "deploy" / "graphs" / "__init__.py"
    if not graphs_init.exists():
        return False

    content = graphs_init.read_text(encoding="utf-8")
    if "holoscan.graphs" not in content or "holoscan.flow_graphs" in content:
        return False

    patched = "\n".join(
        [
            "try:",
            "    from holoscan.graphs import *",
            "except (ImportError, ModuleNotFoundError):",
            "    from holoscan.flow_graphs import *",
            "",
        ]
    )
    graphs_init.write_text(patched, encoding="utf-8")
    return True


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


def ensure_monai_deploy_pipeline_generator(
    install_dir: str = DEFAULT_PIPELINE_GENERATOR_DIR,
    repo_url: str = DEFAULT_PIPELINE_GENERATOR_REPO,
    branch: str = DEFAULT_PIPELINE_GENERATOR_BRANCH,
) -> str:
    """Ensure the MONAI Deploy pipeline-generator CLI is available.

    Pixels does not vendor the upstream MONAI generator. This helper installs
    the generator into the active notebook environment from the MONAI Deploy App
    SDK repository, then returns the `pg` executable path used by
    generate_monai_deploy_app.
    """

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
    _run_command([sys.executable, "-m", "pip", "install", "-e", str(target_dir)])
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
    command = [pg, "gen", model_id, "--format", app_format, "--output", str(deploy_app_dir)]
    if force:
        command.append("-f")
    _run_command(command)

    app_py = deploy_app_dir / "app.py"
    if not app_py.exists():
        raise FileNotFoundError(f"Generated app.py not found in {deploy_app_dir}")
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
    return [str(Path(__file__).resolve().parents[3])]


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
            raise FileNotFoundError(f"app.py not found in deploy app directory: {self.app_dir}")

    def _normalize_input(self, model_input: Any) -> Dict[str, Any]:
        if isinstance(model_input, dict):
            return model_input
        if hasattr(model_input, "to_dict") and callable(getattr(model_input, "to_dict")):
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
        if user_output_dir:
            user_output_dir = os.path.abspath(str(user_output_dir))
        else:
            user_output_dir = os.path.abspath("monai_output")
        Path(user_output_dir).mkdir(parents=True, exist_ok=True)

        python_exe = os.path.join(self.app_dir, ".venv", "bin", "python")
        if not os.path.isfile(python_exe):
            python_exe = sys.executable

        app_py = os.path.join(self.app_dir, "app.py")
        bundle_path = os.path.join(self.app_dir, "model")
        env = os.environ.copy()
        env["BUNDLE_PATH"] = bundle_path
        result = subprocess.run(
            [python_exe, app_py, "-i", image_path, "-o", user_output_dir],
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

        output_files = [str(path) for path in Path(user_output_dir).rglob("*") if path.is_file()]
        return {"output_dir": user_output_dir, "output_files": output_files}


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
        raise FileNotFoundError(f"app.py not found in deploy app directory: {deploy_app_dir}")

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
    pip_requirements = _merge_requirements(app_requirements, DEFAULT_DEPLOY_RUNTIME_REQUIREMENTS)

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
            mlflow_module.pyfunc.log_model(
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
                mlflow_module.register_model(model_uri, registered_model_name or model_name)
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
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"image_path": image_path}
    if output_dir:
        payload["output_dir"] = output_dir
    prompt = _coerce_label_prompt(label_prompt)
    if prompt:
        payload["label_prompt"] = prompt
    if modality:
        payload["modality"] = modality
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
                t.StructField("labels", t.MapType(t.StringType(), t.StringType()), True),
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
                        rows.append(_normalise_prediction_result(model.predict(payload)))
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
