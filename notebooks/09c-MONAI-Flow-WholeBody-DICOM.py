# Databricks notebook source
# MAGIC %md
# MAGIC # DICOM MONAI Whole-Body Segmentation with Pixels and MLflow
# MAGIC
# MAGIC This notebook validates the whole-body DICOM-first MONAI bundle workflow:
# MAGIC read a DICOM series from a Unity Catalog Volume, generate a MONAI Deploy
# MAGIC application from `MONAI/wholeBody_ct_segmentation`, log it as an MLflow
# MAGIC pyfunc model, and write DICOM outputs back to the Volume.

# COMMAND ----------

# DBTITLE 1,Set workflow configuration
import os
import sys
from pathlib import Path


def _add_workspace_src_to_path():
    candidates = []
    try:
        notebook_path = (
            dbutils.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .notebookPath()
            .get()
        )
        notebook_dir = Path("/Workspace" + notebook_path).parent
        candidates.extend([notebook_dir / "src", notebook_dir.parent / "src"])
    except Exception:
        pass
    candidates.extend([Path.cwd() / "src", Path.cwd().parent / "src"])

    for candidate in candidates:
        if (candidate / "dbx" / "pixels" / "modelserving" / "bundles").is_dir():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            return candidate_text
    return None


_add_workspace_src_to_path()


PIXELS_MONAI_VOLUME = ""
PIXELS_MONAI_INPUT_DICOM_SUBPATH = ""
PIXELS_MONAI_OUTPUT_SUBPATH = ""
PIXELS_MONAI_EXPERIMENT_NAME = ""

PIXELS_MONAI_MODEL_ID = "MONAI/wholeBody_ct_segmentation"

# COMMAND ----------

# DBTITLE 1,Initialize Pixels MONAI runtime
from dbx.pixels.modelserving.bundles.monai_runtime import ensure_monai_runtime

ensure_monai_runtime()

# COMMAND ----------

# DBTITLE 1,Configure DICOM MONAI inference
import mlflow
from dbx.pixels.modelserving.bundles.monai_runtime import load_monai_notebook_config

config = load_monai_notebook_config(
    dbutils=globals().get("dbutils"),
    overrides={
        "volume": PIXELS_MONAI_VOLUME,
        "input_dicom_subpath": PIXELS_MONAI_INPUT_DICOM_SUBPATH,
        "output_subpath": PIXELS_MONAI_OUTPUT_SUBPATH,
        "experiment_name": PIXELS_MONAI_EXPERIMENT_NAME,
        "model_id": PIXELS_MONAI_MODEL_ID,
    },
)

volume_root = config.volume_root
input_dicom_dir = config.input_dicom_dir
output_dir = config.output_dir
model_id = config.model_id
experiment_name = config.experiment_name
work_root = config.work_root

print(f"Volume root: {volume_root}")
print(f"DICOM input directory: {input_dicom_dir}")
print(f"Output directory: {output_dir}")
print(f"Model ID: {model_id}")

# COMMAND ----------

# DBTITLE 1,Validate MONAI deploy dependencies
from dbx.pixels.modelserving.bundles.monai_runtime import (
    print_runtime_summary,
    validate_monai_runtime,
)

print_runtime_summary(validate_monai_runtime())

# COMMAND ----------

# DBTITLE 1,Validate inputs and MLflow experiment
if not os.path.isdir(input_dicom_dir):
    raise FileNotFoundError(
        f"DICOM input directory not found: {input_dicom_dir}. Upload a DICOM "
        "series directory to this UC Volume path or adjust input_dicom_subpath."
    )

os.makedirs(output_dir, exist_ok=True)
model_name = model_id.split("/")[-1]
deploy_app_dir = work_root / "deploy_apps" / model_name

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name)

print(f"Experiment: {experiment_name}")
print(f"Deploy app directory: {deploy_app_dir}")

# COMMAND ----------

# DBTITLE 1,Generate MONAI Deploy app in DICOM mode
from dbx.pixels.modelserving.bundles import generate_monai_deploy_app

app_py = deploy_app_dir / "app.py"
if app_py.exists():
    generated_app_dir = str(deploy_app_dir)
    print(f"DICOM deploy app already exists, skipping generation: {generated_app_dir}")
else:
    generated_app_dir = generate_monai_deploy_app(
        model_id=model_id,
        output_dir=str(deploy_app_dir),
        app_format="dicom",
        force=True,
        pipeline_generator_dir=str(work_root / "tools" / "pipeline-generator"),
    )
    print(f"Generated DICOM deploy app: {generated_app_dir}")

# COMMAND ----------

# DBTITLE 1,Log generated app
from dbx.pixels.modelserving.bundles import log_monai_deploy_app

run_id, model_uri = log_monai_deploy_app(
    deploy_app_dir=str(deploy_app_dir),
    input_example_path=input_dicom_dir,
    verbose=True,
)

print(f"Run ID: {run_id}")
print(f"Model URI: {model_uri}")

# COMMAND ----------

# DBTITLE 1,Run single-image inference
import pandas as pd

pyfunc_model = mlflow.pyfunc.load_model(model_uri)
input_df = pd.DataFrame(
    {
        "image_path": [input_dicom_dir],
        "output_dir": [output_dir],
    }
)
preds = pyfunc_model.predict(input_df)

print(f"Output directory: {preds.get('output_dir')}")
print(f"DICOM metadata policy: {preds.get('dicom_metadata_policy')}")
print(f"DICOM metadata repairs: {preds.get('dicom_metadata_repairs')}")
print("Output files:")
for path in preds.get("output_files") or []:
    print(f"  {path}")

# COMMAND ----------

# DBTITLE 1,Verify DICOM outputs
output_files = preds.get("output_files") or []
dicom_outputs = [
    path for path in output_files if path.lower().endswith((".dcm", ".seg.dcm"))
]

if not dicom_outputs:
    raise RuntimeError(f"No DICOM output files found. All output files: {output_files}")

print("Verified DICOM output files:")
for path in dicom_outputs:
    print(f"  {path}")
