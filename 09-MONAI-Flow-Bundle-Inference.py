# Databricks notebook source
# MAGIC %md
# MAGIC # DICOM MONAI Inference with Pixels and MLflow
# MAGIC
# MAGIC This notebook validates the DICOM workflow from the MONAI-Flow integration:
# MAGIC
# MAGIC 1. Read a DICOM series directory from a Unity Catalog Volume.
# MAGIC 2. Generate a MONAI Deploy app in DICOM mode from a public MONAI bundle.
# MAGIC 3. Log that generated app as an MLflow pyfunc model.
# MAGIC 4. Run inference with the DICOM directory as `image_path`.
# MAGIC 5. Verify that DICOM output files are written back to the Volume.
# MAGIC
# MAGIC **Compute requirements:** Databricks Runtime 14.3 LTS ML or later is recommended. GPU compute is recommended for larger bundles. Unity Catalog Volumes are required for durable DICOM input and output paths.
# MAGIC
# MAGIC **Input requirement:** Upload a DICOM series directory to the configured Volume before running the inference cells. The default expects `/Volumes/<catalog>/<schema>/<volume>/monai_flow_demo/input/img_54_bone/`.

# COMMAND ----------

# DBTITLE 1,Initialize Pixels environment
# MAGIC %run ./config/setup

# COMMAND ----------

# DBTITLE 1,Configure DICOM MONAI inference
dbutils.widgets.text(
    "monai_flow_repo_path",
    "",
    label="Workspace path to monai-flow repo, e.g. /Workspace/Users/<you>/monai-flow",
)
dbutils.widgets.text(
    "model_id",
    "MONAI/spleen_ct_segmentation",
    label="Public MONAI model ID for app generation",
)
dbutils.widgets.text(
    "input_dicom_subpath",
    "monai_flow_demo/input/img_54_bone",
    label="DICOM series directory under the configured UC Volume",
)
dbutils.widgets.text(
    "output_subpath",
    "monai_flow_demo/output/dicom_seg",
    label="Output directory under the configured UC Volume",
)
dbutils.widgets.text(
    "experiment_name",
    "",
    label="Optional MLflow experiment path; defaults to /Users/<creator>/pixels-monai-flow",
)

volume_name = dbutils.widgets.get("volume")
volume_root = "/Volumes/" + volume_name.replace(".", "/")

input_dicom_dir = f"{volume_root}/{dbutils.widgets.get('input_dicom_subpath').strip('/')}"
output_dir = f"{volume_root}/{dbutils.widgets.get('output_subpath').strip('/')}"

print(f"Volume root: {volume_root}")
print(f"DICOM input directory: {input_dicom_dir}")
print(f"Output directory: {output_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare MONAI-Flow with deploy support
# MAGIC
# MAGIC This DICOM path uses MONAI-Flow's MONAI Deploy app support. Provide a workspace clone of the MONAI-Flow repository in `monai_flow_repo_path`; the cell below downloads the upstream `pipeline-generator` helper and installs MONAI-Flow with its `deploy` extra.

# COMMAND ----------

# DBTITLE 1,Install MONAI-Flow deploy dependencies
import os
import subprocess
import sys
from pathlib import Path

monai_flow_repo_path = dbutils.widgets.get("monai_flow_repo_path").strip()
if not monai_flow_repo_path:
    raise ValueError(
        "Set the monai_flow_repo_path widget to a workspace clone of the "
        "MONAI-Flow repository before running this DICOM workflow."
    )

repo_path = Path(monai_flow_repo_path)
if not repo_path.exists():
    raise FileNotFoundError(f"MONAI-Flow repo path not found: {repo_path}")

setup_pipeline_generator = repo_path / "scripts" / "setup_pipeline_generator.sh"
if not setup_pipeline_generator.exists():
    raise FileNotFoundError(
        f"Expected MONAI-Flow setup script not found: {setup_pipeline_generator}"
    )

subprocess.check_call([str(setup_pipeline_generator)])
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-e", f"{repo_path}[deploy]"]
)

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Restore configuration after restart
import os
import shutil
import subprocess
from pathlib import Path

import mlflow

volume_name = dbutils.widgets.get("volume")
volume_root = "/Volumes/" + volume_name.replace(".", "/")
input_dicom_dir = f"{volume_root}/{dbutils.widgets.get('input_dicom_subpath').strip('/')}"
output_dir = f"{volume_root}/{dbutils.widgets.get('output_subpath').strip('/')}"
model_id = dbutils.widgets.get("model_id").strip()
monai_flow_repo_path = dbutils.widgets.get("monai_flow_repo_path").strip()
experiment_name = dbutils.widgets.get("experiment_name").strip()

if not experiment_name:
    try:
        creator = spark.conf.get("spark.databricks.clusterUsageTags.creator")
    except Exception:
        creator = "Shared"
    experiment_name = f"/Users/{creator}/pixels-monai-flow"

if not os.path.isdir(input_dicom_dir):
    raise FileNotFoundError(
        f"DICOM input directory not found: {input_dicom_dir}. Upload a DICOM "
        "series directory to this UC Volume path or adjust input_dicom_subpath."
    )

os.makedirs(output_dir, exist_ok=True)

model_name = model_id.split("/")[-1]
work_root = Path("/local_disk0/pixels_monai_flow")
deploy_app_dir = work_root / "deploy_apps" / model_name

print(f"Model ID: {model_id}")
print(f"DICOM input directory: {input_dicom_dir}")
print(f"Output directory: {output_dir}")
print(f"Deploy app directory: {deploy_app_dir}")
print(f"Experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download and generate the DICOM deploy app
# MAGIC
# MAGIC `pg gen` downloads the public MONAI bundle and generates a MONAI Deploy app. The `--format dicom` option makes the generated app accept a DICOM series directory and write DICOM outputs.

# COMMAND ----------

# DBTITLE 1,Generate MONAI Deploy app in DICOM mode
if deploy_app_dir.exists():
    shutil.rmtree(deploy_app_dir)

deploy_app_dir.parent.mkdir(parents=True, exist_ok=True)
subprocess.check_call(
    [
        "pg",
        "gen",
        model_id,
        "--format",
        "dicom",
        "--output",
        str(deploy_app_dir),
        "-f",
    ]
)

assert (deploy_app_dir / "app.py").exists(), f"Generated app.py not found in {deploy_app_dir}"
print(f"Generated DICOM deploy app: {deploy_app_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the generated DICOM app to MLflow
# MAGIC
# MAGIC The logged MLflow model stores the generated app as an artifact. Prediction accepts:
# MAGIC
# MAGIC - `image_path`: DICOM series directory
# MAGIC - `output_dir`: durable Volume path for generated outputs

# COMMAND ----------

# DBTITLE 1,Log generated app
from dbx.pixels.modelserving.bundles import log_monai_flow_bundle

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name)

run_id, model_uri = log_monai_flow_bundle(
    deploy_app_dir=str(deploy_app_dir),
    input_example_path=input_dicom_dir,
    verbose=True,
)

print(f"Run ID: {run_id}")
print(f"Model URI: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run DICOM inference from the UC Volume
# MAGIC
# MAGIC `MonaiFlowBundleTransformer` loads the logged MLflow pyfunc model and runs prediction against the DICOM directory. Outputs are written directly under `output_dir`.

# COMMAND ----------

# DBTITLE 1,Run inference with Pixels transformer
from dbx.pixels.modelserving.bundles import MonaiFlowBundleTransformer

input_df = spark.createDataFrame([(input_dicom_dir,)], ["image_path"])

transformer = MonaiFlowBundleTransformer(
    modelUri=model_uri,
    inputCol="image_path",
    outputCol="monai_result",
    outputDir=output_dir,
    numPartitions=1,
)

result_df = transformer.transform(input_df)
display(result_df)

# COMMAND ----------

# DBTITLE 1,Verify DICOM outputs
result = result_df.collect()[0]["monai_result"]

if result["error"]:
    raise RuntimeError(result["error"])

output_files = result["output_files"] or []
dicom_outputs = [
    path for path in output_files if path.lower().endswith((".dcm", ".seg.dcm"))
]

if not dicom_outputs:
    raise RuntimeError(
        "Inference completed, but no DICOM output files were reported. "
        f"All output files: {output_files}"
    )

print("Verified DICOM output files:")
for path in dicom_outputs:
    print(path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Record for handoff
# MAGIC
# MAGIC Capture the following values for the Databricks handoff:
# MAGIC
# MAGIC - Databricks Runtime version and compute type
# MAGIC - `model_id`
# MAGIC - `input_dicom_dir`
# MAGIC - `model_uri`
# MAGIC - verified DICOM output file paths
