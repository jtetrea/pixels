# Generated MONAI Deploy DICOM Workflow

This experimental branch adds a narrow DICOM-first inference path for public
MONAI Model Zoo bundles. It complements the existing Pixels MONAI Label and
model-serving integration; it does not replace the DICOMweb apps, OHIF workflow,
or existing serving endpoint.

## Workflow

```text
Unity Catalog Volume DICOM series
  -> MONAI pipeline-generator
  -> generated MONAI Deploy app
  -> MLflow pyfunc model
  -> DICOM SEG and optional STL artifacts
  -> Unity Catalog Volume
```

The generated app is packaged with the MLflow model so reload and prediction
exercise the same app that was logged. Durable inputs and outputs stay in Unity
Catalog Volumes; generator and model scratch state stays on ephemeral local
disk.

## Files

| Path | Role |
| --- | --- |
| `notebooks/09a-MONAI-Flow-Spleen-DICOM.py` | Spleen regression notebook. |
| `notebooks/09b-MONAI-Flow-Pancreas-DICOM.py` | Pancreas and pancreatic-tumor notebook. |
| `notebooks/09c-MONAI-Flow-WholeBody-DICOM.py` | Whole-body high-memory GPU notebook. |
| `src/dbx/pixels/modelserving/bundles/monai_flow.py` | Pipeline generation, generated-app patching, MLflow logging, pyfunc prediction, and deidentified-DICOM preparation. |
| `src/dbx/pixels/modelserving/bundles/monai_runtime.py` | Databricks AI Runtime dependency bootstrap, configuration, and runtime checks. |
| `tests/dbx/test_monai_flow_bundle.py` | CPU-safe contract and generated-app regression tests. |

## Supported API

```python
from dbx.pixels.modelserving.bundles import (
    generate_monai_deploy_app,
    log_monai_deploy_app,
)

deploy_app_dir = generate_monai_deploy_app(
    model_id="MONAI/spleen_ct_segmentation",
    output_dir="/tmp/monai-flow/deploy_apps/spleen",
    app_format="dicom",
)

run_id, model_uri = log_monai_deploy_app(
    deploy_app_dir=deploy_app_dir,
    input_example_path="/Volumes/catalog/schema/volume/input/series",
)
```

Prediction accepts a one-row pandas DataFrame with `image_path` and
`output_dir`. The result reports `output_files`, `output_dir`, the DICOM metadata
policy, and any safe metadata repairs.

## Generated-App Patch Boundary

Pipeline-generator output can contain too few DICOM SEG
`SegmentDescription` objects for multi-label bundles. Pixels replaces those
generated descriptions with non-background labels from the public bundle
metadata.

This branch deliberately does not modify the model's preprocessing, network,
post-processing, or numerical behavior. In particular, it does not bypass the
whole-body bundle's softmax to make it fit on a smaller GPU. Resource mapping is
part of validation, not a reason to silently change the bundle.

`tests/dbx/test_monai_flow_bundle.py` statically guards this boundary so the
former inference-altering helper names and allocator override cannot be
reintroduced unnoticed.

The default `deidentified_safe` input policy creates an ephemeral DICOM copy
when highdicom-required Type 2 fields are missing. Missing values are filled
with zero-length attributes; redacted values are not reconstructed. Use
`strict` to disable this behavior.

## Runtime Contract

The validated AI Runtime v5 stack installs the MONAI Deploy pieces without
upgrading Databricks-managed PyTorch, CUDA, or MLflow:

```text
monai==1.5.1
monai-deploy-app-sdk==3.5.0
holoscan==3.10.0
holoscan-cu12==3.10.0
protobuf>=5.26.1,<6.0dev
grpcio>=1.67.1,<1.68
grpcio-status>=1.67.1,<1.68
tritonclient[http,grpc]==2.60.0
mlflow>=3,<4
```

The pipeline-generator Python constraint is relaxed for Databricks Python 3.12
during installation. `holoscan-cli` and `databricks-connect` are intentionally
not installed into the notebook environment.

## Validation Matrix

| Model | Local generated DICOM | Databricks DICOM | Resource decision |
| --- | --- | --- | --- |
| `MONAI/spleen_ct_segmentation` | Pass | Pass | AI Runtime v5 `GPU_1xA10`. |
| `MONAI/pancreas_ct_dints_segmentation` | Pass | Pass | AI Runtime v5 `GPU_1xA10`. |
| `MONAI/wholeBody_ct_segmentation` | Pass | Pass | A10 reaches unmodified softmax and OOMs; AI Runtime v5 `GPU_1xH100` passes unchanged inference/post-processing with a 104-segment, non-empty DICOM SEG. |

The current notebooks are single-process. Selecting `GPU_8xH100` alone would
not make the generated MONAI Deploy app multi-GPU; that would require an
explicit distributed design.

The shared technical DICOM fixture proves governed input, app generation,
MLflow round-trip, DICOM SEG writing, and durable output. Task-relevant DICOM
series and labels are still needed for semantic validation.

The completed whole-body acceptance gate additionally downloads the durable
SEG and verifies Segmentation Storage SOP Class identity, 104 unique named
segments, readable decoded pixels, and nonzero segmentation content. That
model-independent verifier currently lives in MONAI Flow and is a candidate for
the reusable upstream test surface.

## Local Tests

Run the focused branch tests with the source tree first on `PYTHONPATH`:

```bash
PYTHONPATH=src pytest -q \
  tests/dbx/test_monai_flow_bundle.py \
  tests/dbx/test_dependency.py
```

## Contribution Boundary

A contribution should include the reusable helper modules, dependency/runtime
contract, model-agnostic notebook pattern, DICOM SEG acceptance checks, and
CPU-safe tests. Workspace paths, run IDs, private fixtures, and internal staging
utilities belong outside Pixels. The canonical spleen, pancreas, and whole-body
paths are now Databricks-validated without modifying bundle inference behavior.
