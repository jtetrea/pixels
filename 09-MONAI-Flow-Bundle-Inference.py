# Databricks notebook source
# MAGIC %md
# MAGIC # MONAI Bundle Inference with Pixels and MLflow
# MAGIC
# MAGIC This notebook demonstrates a minimal MONAI bundle inference workflow using Pixels, Unity Catalog Volumes, and MLflow.
# MAGIC
# MAGIC **What this notebook does**
# MAGIC
# MAGIC 1. Uses the standard Pixels setup widgets for catalog, schema, table, and volume configuration.
# MAGIC 2. Installs the optional `monai-flow` package from a workspace path when needed.
# MAGIC 3. Logs a MONAI bundle as an MLflow pyfunc model.
# MAGIC 4. Runs inference against an input image stored in a Unity Catalog Volume.
# MAGIC 5. Writes model outputs back to the same governed Volume.
# MAGIC
# MAGIC **Compute requirements:** Databricks Runtime 14.3 LTS ML or later is recommended. A GPU cluster is recommended for larger bundles; the default spleen CT bundle is suitable for smoke testing.

# COMMAND ----------

# DBTITLE 1,Initialize Pixels environment
# MAGIC %run ./config/setup

# COMMAND ----------

# DBTITLE 1,Configure MONAI bundle inference
dbutils.widgets.text(
    "monai_flow_repo_path",
    "",
    label="Optional workspace path to monai-flow repo, e.g. /Workspace/Users/<you>/monai-flow",
)
dbutils.widgets.text(
    "bundle_name",
    "spleen_ct_segmentation",
    label="MONAI bundle name",
)
dbutils.widgets.text(
    "input_subpath",
    "monai_flow_demo/input/spleen_10.nii.gz",
    label="Input path under the configured UC Volume",
)
dbutils.widgets.text(
    "output_subpath",
    "monai_flow_demo/output/spleen_ct_segmentation",
    label="Output directory under the configured UC Volume",
)
dbutils.widgets.text(
    "bundle_subpath",
    "monai_flow_demo/bundles",
    label="Bundle download directory under the configured UC Volume",
)
dbutils.widgets.text(
    "experiment_name",
    "",
    label="Optional MLflow experiment path; defaults to /Users/<creator>/pixels-monai-flow",
)

volume_name = dbutils.widgets.get("volume")
volume_root = "/Volumes/" + volume_name.replace(".", "/")

input_path = f"{volume_root}/{dbutils.widgets.get('input_subpath').strip('/')}"
output_dir = f"{volume_root}/{dbutils.widgets.get('output_subpath').strip('/')}"
bundle_download_dir = f"{volume_root}/{dbutils.widgets.get('bundle_subpath').strip('/')}"

print(f"Volume root: {volume_root}")
print(f"Input path: {input_path}")
print(f"Output directory: {output_dir}")
print(f"Bundle download directory: {bundle_download_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install optional MONAI-Flow dependency
# MAGIC
# MAGIC Pixels keeps MONAI-Flow optional so the base accelerator remains lightweight. If `monai-flow` is already installed on the cluster, leave `monai_flow_repo_path` blank. If you cloned the MONAI-Flow repository into the workspace, provide its workspace path in the widget above.

# COMMAND ----------

# DBTITLE 1,Install monai-flow when a workspace repo path is provided
import importlib.util
import subprocess
import sys

monai_flow_repo_path = dbutils.widgets.get("monai_flow_repo_path").strip()

if monai_flow_repo_path:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", monai_flow_repo_path]
    )
elif importlib.util.find_spec("monai_flow") is None:
    raise ImportError(
        "monai_flow is not installed. Set the monai_flow_repo_path widget to a "
        "workspace clone of the MONAI-Flow repository, then rerun this cell."
    )
else:
    print("monai_flow is already available on this cluster.")

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Restore widget values after restart
import os

volume_name = dbutils.widgets.get("volume")
volume_root = "/Volumes/" + volume_name.replace(".", "/")
bundle_name = dbutils.widgets.get("bundle_name")
input_path = f"{volume_root}/{dbutils.widgets.get('input_subpath').strip('/')}"
output_dir = f"{volume_root}/{dbutils.widgets.get('output_subpath').strip('/')}"
bundle_download_dir = f"{volume_root}/{dbutils.widgets.get('bundle_subpath').strip('/')}"
experiment_name = dbutils.widgets.get("experiment_name").strip()

if not experiment_name:
    try:
        creator = spark.conf.get("spark.databricks.clusterUsageTags.creator")
    except Exception:
        creator = "Shared"
    experiment_name = f"/Users/{creator}/pixels-monai-flow"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(bundle_download_dir, exist_ok=True)

if not os.path.exists(input_path):
    raise FileNotFoundError(
        f"Input file not found: {input_path}. Upload a sample image to the UC "
        "Volume path or adjust the input_subpath widget."
    )

print(f"Bundle: {bundle_name}")
print(f"Experiment: {experiment_name}")
print(f"Input: {input_path}")
print(f"Output: {output_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the MONAI bundle to MLflow
# MAGIC
# MAGIC The bundle weights are downloaded at runtime into the configured Unity Catalog Volume. They are not stored in the Pixels repository.

# COMMAND ----------

# DBTITLE 1,Log bundle
import mlflow

from dbx.pixels.modelserving.bundles import log_monai_flow_bundle

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name)

run_id, model_uri = log_monai_flow_bundle(
    bundle_name=bundle_name,
    download=True,
    bundle_download_dir=bundle_download_dir,
    input_example_path=input_path,
    verbose=True,
)

print(f"Run ID: {run_id}")
print(f"Model URI: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run inference with the Pixels transformer
# MAGIC
# MAGIC `MonaiFlowBundleTransformer` applies a logged MONAI-Flow MLflow model to paths in a Spark DataFrame. For this smoke test, the DataFrame contains one input image.

# COMMAND ----------

# DBTITLE 1,Run inference
from dbx.pixels.modelserving.bundles import MonaiFlowBundleTransformer

input_df = spark.createDataFrame([(input_path,)], ["image_path"])

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

# DBTITLE 1,Verify persisted outputs
result = result_df.collect()[0]["monai_result"]

if result["error"]:
    raise RuntimeError(result["error"])

if not result["output_files"]:
    raise RuntimeError("Inference completed without output files.")

print("Output directory:", result["output_dir"])
print("Output files:")
for path in result["output_files"]:
    print(path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC - Use this notebook first with `spleen_ct_segmentation` and a small CT NIfTI file.
# MAGIC - Record the Databricks Runtime version, compute type, model URI, and output file paths.
# MAGIC - After the direct bundle path is validated, extend testing to DICOM SEG output and larger bundles.
