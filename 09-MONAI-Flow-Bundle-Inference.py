# Databricks notebook source
# MAGIC %md
# MAGIC # DICOM MONAI Inference with Pixels and MLflow
# MAGIC
# MAGIC This notebook validates the DICOM workflow from the Pixels MONAI integration:
# MAGIC
# MAGIC 1. Read a DICOM series directory from a Unity Catalog Volume.
# MAGIC 2. Generate a MONAI Deploy app in DICOM mode from a public MONAI bundle.
# MAGIC 3. Log that generated app as an MLflow pyfunc model.
# MAGIC 4. Run inference with the DICOM directory as `image_path`.
# MAGIC 5. Verify that DICOM output files are written back to the Volume.
# MAGIC
# MAGIC **Compute requirements:** GPU serverless (A10G recommended) or DBR 14.3+ ML with GPU.
# MAGIC Unity Catalog Volumes are required for durable DICOM input/output paths.
# MAGIC
# MAGIC **Input requirement:** Upload a DICOM series directory to the configured Volume before
# MAGIC running the inference cells.

# COMMAND ----------

# DBTITLE 1,Initialize Pixels environment
import subprocess
import sys

# GPU serverless preinstalls most of this stack. Install only the exact packages
# this workflow needs, without asking pip to resolve or upgrade Databricks-managed
# packages such as MLflow and Databricks Connect.
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        "monai-deploy-app-sdk==3.5.0",
        "holoscan==4.0.0",
        "holoscan-cu12==4.0.0",
        "colorama>=0.4.1",
        "typeguard>=3.0.0",
        "pytorch-ignite>=0.4",
        "numpy-stl>=3.0",
        "trimesh",
    ]
)

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configure DICOM MONAI inference
dbutils.widgets.text(
    "volume",
    "catalog.schema.volume_name",
    label="Configured UC Volume (catalog.schema.volume)",
)
dbutils.widgets.text(
    "model_id",
    "MONAI/spleen_ct_segmentation",
    label="Public MONAI model ID for app generation",
)
dbutils.widgets.text(
    "input_dicom_subpath",
    "dicom_input/series_dir",
    label="DICOM series directory under the configured UC Volume",
)
dbutils.widgets.text(
    "output_subpath",
    "monai_flow_output/",
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
# MAGIC ## Validate environment
# MAGIC
# MAGIC Confirm the MONAI Deploy / Holoscan versions and verify GPU availability.

# COMMAND ----------

# DBTITLE 1,Validate MONAI deploy dependencies
# Validate environment after restart
import monai
import monai.deploy.conditions
import holoscan
import importlib.metadata as metadata
import mlflow
import torch

print(
    f"monai {monai.__version__}, "
    f"monai-deploy {metadata.version('monai-deploy-app-sdk')}, "
    f"holoscan-cu12 {metadata.version('holoscan-cu12')}, "
    f"holoscan {holoscan.__version__}, torch {torch.__version__}, "
    f"mlflow {mlflow.__version__}"
)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# COMMAND ----------

# DBTITLE 1,Restore configuration after restart
import os
from pathlib import Path

import mlflow

volume_name = dbutils.widgets.get("volume")
volume_root = "/Volumes/" + volume_name.replace(".", "/")
input_dicom_dir = f"{volume_root}/{dbutils.widgets.get('input_dicom_subpath').strip('/')}"
output_dir = f"{volume_root}/{dbutils.widgets.get('output_subpath').strip('/')}"
model_id = dbutils.widgets.get("model_id").strip()
experiment_name = dbutils.widgets.get("experiment_name").strip()

print(f"Volume root: {volume_root}")
print(f"DICOM input directory: {input_dicom_dir}")
print(f"Output directory: {output_dir}")

if not experiment_name:
    try:
        creator = spark.sql("SELECT current_user()").first()[0]
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
# Use /tmp for ephemeral work (pipeline-generator install, app generation workspace)
# The generated app is also written to the UC Volume via deploy_app_dir
work_root = Path("/tmp/pixels_monai_flow")
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
# MAGIC `generate_monai_deploy_app` installs the MONAI pipeline generator if needed,
# MAGIC downloads the public MONAI bundle, and creates a MONAI Deploy app. The `dicom`
# MAGIC format makes the generated app accept a DICOM series directory and write DICOM outputs.

# COMMAND ----------

# DBTITLE 1,Generate MONAI Deploy app in DICOM mode
from dbx.pixels.modelserving.bundles import generate_monai_deploy_app

# Skip generation if the app already exists (idempotent)
app_py = deploy_app_dir / "app.py"
if app_py.exists():
    generated_app_dir = str(deploy_app_dir)
    print(f"DICOM deploy app already exists, skipping generation: {generated_app_dir}")
else:
    pg_dir = "/tmp/pixels_monai_flow/tools/pipeline-generator"
    generated_app_dir = generate_monai_deploy_app(
        model_id=model_id,
        output_dir=str(deploy_app_dir),
        app_format="dicom",
        force=False,
        pipeline_generator_dir=pg_dir,
    )
    print(f"Generated DICOM deploy app: {generated_app_dir}")

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
from dbx.pixels.modelserving.bundles import log_monai_deploy_app

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name)

run_id, model_uri = log_monai_deploy_app(
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
# MAGIC Load the logged MLflow pyfunc model directly and run prediction with a pandas
# MAGIC DataFrame. This follows the same pattern as the `monai_flow_demo` notebook:
# MAGIC no Spark overhead needed for single-image inference.

# COMMAND ----------

# DBTITLE 1,Run single-image inference
import pandas as pd

pyfunc_model = mlflow.pyfunc.load_model(model_uri)

input_df = pd.DataFrame({"image_path": [input_dicom_dir], "output_dir": [output_dir]})
preds = pyfunc_model.predict(input_df)

print(f"Output files: {preds.get('output_files')}")
if preds.get("error"):
    print(f"Error: {preds['error']}")

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

# COMMAND ----------

# DBTITLE 1,Visualize DICOM segmentation overlay
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# Load input DICOM series (sorted by InstanceNumber or ImagePositionPatient)
dicom_files = sorted(
    [str(Path(input_dicom_dir) / f) for f in os.listdir(input_dicom_dir)
     if f.lower().endswith(".dcm")]
)
dicom_images = [pydicom.dcmread(f) for f in dicom_files]
# Sort by slice position for correct z-ordering
try:
    dicom_images.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
except (AttributeError, IndexError):
    dicom_images.sort(key=lambda ds: int(getattr(ds, "InstanceNumber", 0)))

dicom_pixel_arrays = np.stack([ds.pixel_array for ds in dicom_images])
print(f"Input CT volume: {dicom_pixel_arrays.shape} (slices, H, W)")

# Load DICOM SEG output
seg_file = next((f for f in output_files if f.lower().endswith(".dcm")), None)
if not seg_file:
    raise FileNotFoundError("No DICOM SEG file found in outputs.")

seg_ds = pydicom.dcmread(seg_file)
seg_arr = seg_ds.pixel_array
print(f"Segmentation array: {seg_arr.shape}, unique values: {np.unique(seg_arr)}")

# Validate segmentation is non-zero
total_seg_voxels = np.count_nonzero(seg_arr)
if total_seg_voxels == 0:
    print("\nWARNING: Segmentation is entirely zero - no structure was segmented.")
    print("This may indicate the model did not detect the target organ in this scan.")
else:
    print(f"\nSegmentation contains {total_seg_voxels:,} non-zero voxels")

    # Find slices with segmentation.
    # seg_arr may be (num_seg_frames, H, W); find per-frame nonzero counts.
    if seg_arr.ndim == 3:
        seg_per_slice = np.array([np.count_nonzero(seg_arr[i]) for i in range(seg_arr.shape[0])])
    else:
        seg_per_slice = np.array([np.count_nonzero(seg_arr)])

    active_frames = np.where(seg_per_slice > 0)[0]
    print(f"  Segmentation present in {len(active_frames)}/{seg_arr.shape[0]} frames")
    print(f"  Frame range with segmentation: [{active_frames[0]}, {active_frames[-1]}]")

    # Pick the middle frame that has segmentation.
    mid_frame_idx = active_frames[len(active_frames) // 2]
    mid_seg_slice = seg_arr[mid_frame_idx]

    # Calculate area for the middle slice.
    pixel_count = np.count_nonzero(mid_seg_slice)
    # Try to get pixel spacing for real-world area
    try:
        ps = dicom_images[0].PixelSpacing
        pixel_area_mm2 = float(ps[0]) * float(ps[1])
        area_mm2 = pixel_count * pixel_area_mm2
        area_cm2 = area_mm2 / 100.0
        area_label = f"area: {area_cm2:.2f} cm2"
        print(f"\n  Middle seg frame {mid_frame_idx}: {pixel_count:,} pixels")
        print(f"  Cross-sectional area: {area_mm2:.1f} mm2 ({area_cm2:.2f} cm2)")
    except (AttributeError, IndexError):
        area_label = "area: unavailable"
        print(
            f"\n  Middle seg frame {mid_frame_idx}: {pixel_count:,} pixels "
            "(no pixel spacing available)"
        )

    # Map seg frames to CT slices.
    # DICOM SEG frames may reference source slices via ReferencedSOPInstanceUID
    # Simple approach: if seg has fewer slices, try to align via PerFrameFunctionalGroupsSequence
    ct_slice_for_seg = None
    try:
        pffg = seg_ds.PerFrameFunctionalGroupsSequence
        frame_item = pffg[mid_frame_idx]
        ref_sop = (
            frame_item.DerivationImageSequence[0]
            .SourceImageSequence[0]
            .ReferencedSOPInstanceUID
        )
        # Find matching CT slice
        for ct_idx, ds in enumerate(dicom_images):
            if ds.SOPInstanceUID == ref_sop:
                ct_slice_for_seg = ct_idx
                break
    except (AttributeError, IndexError, KeyError):
        # Fallback: assume seg frames map proportionally to CT volume
        ct_slice_for_seg = int(mid_frame_idx * dicom_pixel_arrays.shape[0] / seg_arr.shape[0])

    if ct_slice_for_seg is None:
        ct_slice_for_seg = dicom_pixel_arrays.shape[0] // 2

    # Visualize.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: CT slice alone
    axes[0].imshow(dicom_pixel_arrays[ct_slice_for_seg], cmap="gray")
    axes[0].set_title(f"CT Slice {ct_slice_for_seg}")
    axes[0].axis("off")

    # Middle: Segmentation mask alone
    axes[1].imshow(mid_seg_slice, cmap="hot", interpolation="nearest")
    axes[1].set_title(f"Seg Frame {mid_frame_idx} ({pixel_count:,} px)")
    axes[1].axis("off")

    # Right: Overlay
    axes[2].imshow(dicom_pixel_arrays[ct_slice_for_seg], cmap="gray")
    masked = np.ma.masked_where(mid_seg_slice == 0, mid_seg_slice)
    axes[2].imshow(masked, cmap="autumn", alpha=0.5)
    axes[2].set_title(f"Overlay ({area_label})")
    axes[2].axis("off")

    plt.suptitle(f"{model_id} - Middle Segmentation Slice", fontsize=13)
    plt.tight_layout()
    plt.show()
