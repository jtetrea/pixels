# DICOM MONAI MLflow Branch Handoff

Branch: `feature/dicom-monai-mlflow`  
Base: `main`

## Summary

This fork adds a minimal DICOM-first MONAI Deploy + MLflow validation path to
Pixels. The workflow generates a MONAI Deploy app from a public MONAI bundle,
logs the generated app as an MLflow pyfunc model, runs inference on a DICOM
series stored in a Unity Catalog Volume, writes DICOM outputs back to the
Volume, and visualizes the produced DICOM SEG output.

The first validated path is single-series inference. Spark batch inference is
included as a helper class but is not used by the notebook until the single-image
path is stable.

## File Delta From `main`

| File | Change | Purpose |
| --- | --- | --- |
| `09-MONAI-Flow-Bundle-Inference.py` | Added | Databricks notebook for UC Volume DICOM input, MONAI Deploy app generation, MLflow logging, single-series pyfunc inference, DICOM output verification, and DICOM SEG visualization. |
| `dbx/pixels/modelserving/bundles/monai_flow.py` | Added | Self-contained Pixels helpers for MONAI Deploy app generation, MLflow pyfunc logging, generated app prediction, output normalization, and optional Spark transformer support. |
| `dbx/pixels/modelserving/bundles/__init__.py` | Added | Exports the MONAI helper APIs from `dbx.pixels.modelserving.bundles`. |
| `tests/dbx/test_monai_flow_bundle.py` | Added | Focused unit coverage for payload construction, result normalization, deploy app model input handling, runtime pins, generator compatibility handling, and transformer configuration. |
| `README.md` | Modified | Adds a short section describing the DICOM MONAI MLflow notebook and adds third-party acknowledgements for MONAI/MLflow-related dependencies. |
| `DICOM_MONAI_MLFLOW_HANDOFF.md` | Added | This brief Databricks-facing branch summary for email/PR context. |

Code/notebook delta before this handoff note: 5 files changed, about 1,076
insertions and 2 deletions.

## Runtime Notes

- Target compute: Databricks GPU serverless or GPU ML Runtime.
- Input: one DICOM CT series directory under `/Volumes/<catalog>/<schema>/<volume>/...`.
- Default model: `MONAI/spleen_ct_segmentation`.
- The notebook installs only the MONAI runtime packages needed for this flow and
  avoids resolving/upgrading Databricks-managed packages.
- MONAI Deploy is pinned to `monai-deploy-app-sdk==3.5.0`.
- Holoscan is pinned to `holoscan==4.0.0` / `holoscan-cu12==4.0.0` because
  Holoscan 4.1+ removes the `holoscan.graphs` API used by MONAI Deploy 3.5.
- Model weights and generated apps are not committed. They are downloaded or
  generated at notebook runtime.

## Current State

Validated directionally on Databricks with single-series DICOM inference and
DICOM SEG visualization. The latest branch includes a follow-up installer
change to avoid pip dependency resolver conflicts on Databricks GPU serverless;
that exact installer change should be re-run in Databricks before opening a
non-draft PR.

## Open Questions For Review

- Should this remain a notebook-first optional integration, or move toward a
  formal `databricks-pixels[monai]` package extra?
- Should the module be renamed from `monai_flow.py` to a more native Pixels name
  such as `monai_deploy.py` before PR review?
- Should Spark batch inference remain in this branch, or be split into a later
  follow-up after single-series validation is accepted?
- Should the pipeline-generator dependency be downloaded at runtime as implemented,
  or vendored/pinned another way for stricter reproducibility?
