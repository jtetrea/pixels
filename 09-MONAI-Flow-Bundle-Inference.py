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

# DBTITLE 1,Initialize Pixels MONAI runtime
import importlib
import os
from pathlib import Path

# Serverless GPU owns CUDA and PyTorch. Keep MLflow and Databricks Connect
# inside the serverless-gpu package contract while installing only the MONAI
# Deploy stack and DICOM helpers needed by this workflow.
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
    "monai>=1.5",
    "monai-deploy-app-sdk==3.5.0",
    "holoscan==4.0.0",
    "holoscan-cu12==4.0.0",
    "colorama>=0.4.1",
    "typeguard>=3.0.0",
    "pydicom>=2.3",
    "highdicom",
    "pyjpegls",
    "nibabel",
    "SimpleITK>=2.0",
    "scipy",
    "scikit-image",
    "Pillow",
    "pytorch-ignite>=0.4",
    "numpy-stl>=3.0",
    "trimesh",
]
TRITON_CLIENT_PACKAGES = [
    "protobuf>=5.26.1,<6.0dev",
    "grpcio>=1.67.1,<1.68",
    "grpcio-status>=1.67.1,<1.68",
    "tritonclient[http,grpc]==2.60.0",
]


def _pip_install(packages, *, no_deps):
    from pip._internal.cli.main import main as pip_main

    args = ["install", "-q", "--disable-pip-version-check"]
    if no_deps:
        args.append("--no-deps")
    args.extend(packages)
    exit_code = pip_main(args)
    if exit_code:
        raise RuntimeError(f"pip install failed with exit code {exit_code}: {packages}")


# Run pip in-process so a stale holoscan-cu12 .pth from an earlier failed
# install cannot emit startup warnings before wheel-axle-runtime is available.
_pip_install(BOOTSTRAP_PACKAGES, no_deps=True)
_pip_install(MONAI_DEPLOY_PACKAGES, no_deps=True)
_pip_install(DATABRICKS_SERVERLESS_PACKAGES + TRITON_CLIENT_PACKAGES, no_deps=False)
importlib.invalidate_caches()

# holoscan-cu12 uses a wheel-axle .pth hook to finish installing shared
# libraries. Databricks notebooks keep the current Python process alive after
# pip installs, so run the hook explicitly instead of relying on interpreter
# startup to process the new .pth file.
try:
    import site
    import wheel_axle.runtime

    site_dirs = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        site_dirs.append(user_site)
    for site_dir in site_dirs:
        for pth_path in Path(site_dir).glob("holoscan_cu12-*.pth"):
            wheel_axle.runtime.finalize(str(pth_path))
except Exception as exc:
    raise RuntimeError("Failed to activate holoscan-cu12 wheel-axle runtime") from exc

importlib.invalidate_caches()

# COMMAND ----------

# DBTITLE 1,Configure DICOM MONAI inference
import mlflow

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
    "monai_flow_output",
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
model_id = dbutils.widgets.get("model_id").strip()
experiment_name = dbutils.widgets.get("experiment_name").strip()
work_root = Path("/tmp/pixels_monai_flow")

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
import importlib.metadata as metadata

import holoscan
import monai
import monai.deploy.conditions
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

# DBTITLE 1,Validate inputs and MLflow experiment
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
input_df = pd.DataFrame({"image_path": [input_dicom_dir], "output_dir": [output_dir]})
preds = pyfunc_model.predict(input_df)

print(f"Output directory: {preds.get('output_dir')}")
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
