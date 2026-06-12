# Databricks notebook source
# MAGIC %md
# MAGIC # DICOM MONAI Inference with Pixels and MLflow
# MAGIC
# MAGIC This notebook validates the DICOM workflow from the Pixels MONAI integration:
# MAGIC
# MAGIC 1. Read a DICOM series directory from a Unity Catalog Volume.
# MAGIC 2. Install the MONAI Deploy packages needed by serverless GPU without changing Databricks-managed CUDA/PyTorch packages.
# MAGIC 3. Generate a MONAI Deploy app in DICOM mode from a public MONAI bundle.
# MAGIC 4. Log that generated app as an MLflow pyfunc model.
# MAGIC 5. Run inference with the DICOM directory as `image_path`.
# MAGIC 6. Verify that DICOM output files are written back to the Volume.
# MAGIC
# MAGIC **Compute requirements:** GPU serverless (A10G recommended) or DBR 14.3+ ML with GPU.
# MAGIC Unity Catalog Volumes are required for durable DICOM input/output paths.
# MAGIC
# MAGIC **Input requirement:** Upload a DICOM series directory to the configured Volume before
# MAGIC running the inference cells.

# COMMAND ----------

# DBTITLE 1,Set workflow configuration
import os
from pathlib import Path

# Set these four values before running all cells. Databricks jobs can also pass
# them as task parameters, and advanced users can set the matching environment
# variables before this notebook runs.
PIXELS_MONAI_VOLUME = ""
PIXELS_MONAI_INPUT_DICOM_SUBPATH = ""
PIXELS_MONAI_OUTPUT_SUBPATH = ""
PIXELS_MONAI_EXPERIMENT_NAME = ""

PIXELS_MONAI_MODEL_ID = "MONAI/spleen_ct_segmentation"

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

volume_name = config.volume_name
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

# MAGIC %md
# MAGIC ## Validate environment
# MAGIC
# MAGIC Confirm the MONAI Deploy / Holoscan versions and verify GPU availability before generating the app.

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

# MAGIC %md
# MAGIC ## Download and generate the DICOM deploy app
# MAGIC
# MAGIC `generate_monai_deploy_app` installs the MONAI pipeline generator if needed,
# MAGIC downloads the public MONAI bundle, and creates a MONAI Deploy app. The `dicom`
# MAGIC format makes the generated app accept a DICOM series directory and write DICOM outputs.

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

# MAGIC %md
# MAGIC ## Log the generated DICOM app to MLflow
# MAGIC
# MAGIC The logged MLflow model stores the generated app as an artifact. Prediction
# MAGIC accepts:
# MAGIC
# MAGIC - `image_path`: DICOM series directory
# MAGIC - `output_dir`: durable Volume path for generated outputs
# MAGIC
# MAGIC The pyfunc wrapper uses `deidentified_safe` DICOM metadata handling by
# MAGIC default, filling missing redacted Type 2 source fields as zero-length values
# MAGIC in a temporary copy before DICOM SEG writing.

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

# MAGIC %md
# MAGIC ## Run DICOM inference from the UC Volume
# MAGIC
# MAGIC Load the logged MLflow pyfunc model directly and run prediction with a pandas
# MAGIC DataFrame. Spark batch inference can be added after the single-series path is
# MAGIC validated in the target Databricks workspace.

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

# COMMAND ----------

# DBTITLE 1,Visualize DICOM segmentation overlay
import numpy as np
import pydicom
import matplotlib.pyplot as plt

# Load input DICOM series, sorted by slice position when available.
dicom_files = sorted(
    [str(Path(input_dicom_dir) / f) for f in os.listdir(input_dicom_dir) if f.lower().endswith(".dcm")]
)
dicom_images = [pydicom.dcmread(f) for f in dicom_files]
try:
    dicom_images.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
except (AttributeError, IndexError):
    dicom_images.sort(key=lambda ds: int(getattr(ds, "InstanceNumber", 0)))

dicom_pixel_arrays = np.stack([ds.pixel_array for ds in dicom_images])
print(f"Input CT volume: {dicom_pixel_arrays.shape} (slices, H, W)")

# Prefer DICOM SEG outputs when the file naming makes that explicit.
seg_file = next((f for f in output_files if f.lower().endswith(".seg.dcm")), None)
seg_file = seg_file or next((f for f in output_files if f.lower().endswith(".dcm")), None)
if not seg_file:
    raise FileNotFoundError("No DICOM SEG file found in outputs.")

seg_ds = pydicom.dcmread(seg_file)
seg_arr = seg_ds.pixel_array
print(f"Segmentation array: {seg_arr.shape}, unique values: {np.unique(seg_arr)}")

total_seg_voxels = np.count_nonzero(seg_arr)
if total_seg_voxels == 0:
    print("\nWARNING: Segmentation is entirely zero - no structure was segmented.")
    print("This may indicate the model did not detect the target organ in this scan.")
else:
    print(f"\nSegmentation contains {total_seg_voxels:,} non-zero voxels")

    if seg_arr.ndim == 3:
        seg_per_slice = np.array([np.count_nonzero(seg_arr[i]) for i in range(seg_arr.shape[0])])
    else:
        seg_per_slice = np.array([np.count_nonzero(seg_arr)])

    active_frames = np.where(seg_per_slice > 0)[0]
    print(f"  Segmentation present in {len(active_frames)}/{seg_arr.shape[0]} frames")
    print(f"  Frame range with segmentation: [{active_frames[0]}, {active_frames[-1]}]")

    mid_frame_idx = active_frames[len(active_frames) // 2]
    mid_seg_slice = seg_arr[mid_frame_idx]
    pixel_count = np.count_nonzero(mid_seg_slice)

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
        print(f"\n  Middle seg frame {mid_frame_idx}: {pixel_count:,} pixels")

    ct_slice_for_seg = None
    try:
        frame_item = seg_ds.PerFrameFunctionalGroupsSequence[mid_frame_idx]
        ref_sop = (
            frame_item.DerivationImageSequence[0]
            .SourceImageSequence[0]
            .ReferencedSOPInstanceUID
        )
        for ct_idx, ds in enumerate(dicom_images):
            if ds.SOPInstanceUID == ref_sop:
                ct_slice_for_seg = ct_idx
                break
    except (AttributeError, IndexError, KeyError):
        ct_slice_for_seg = int(mid_frame_idx * dicom_pixel_arrays.shape[0] / seg_arr.shape[0])

    if ct_slice_for_seg is None:
        ct_slice_for_seg = dicom_pixel_arrays.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(dicom_pixel_arrays[ct_slice_for_seg], cmap="gray")
    axes[0].set_title(f"CT Slice {ct_slice_for_seg}")
    axes[0].axis("off")

    axes[1].imshow(mid_seg_slice, cmap="hot", interpolation="nearest")
    axes[1].set_title(f"Seg Frame {mid_frame_idx} ({pixel_count:,} px)")
    axes[1].axis("off")

    axes[2].imshow(dicom_pixel_arrays[ct_slice_for_seg], cmap="gray")
    masked = np.ma.masked_where(mid_seg_slice == 0, mid_seg_slice)
    axes[2].imshow(masked, cmap="autumn", alpha=0.5)
    axes[2].set_title(f"Overlay ({area_label})")
    axes[2].axis("off")

    plt.suptitle(f"{model_id} - Middle Segmentation Slice", fontsize=13)
    plt.tight_layout()
    plt.show()
