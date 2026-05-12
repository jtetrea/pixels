from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, Iterator, Optional

import pandas as pd

try:
    from pyspark.ml.pipeline import Transformer
except ImportError:  # pragma: no cover - local unit-test fallback when Spark is unavailable

    class Transformer:  # type: ignore[no-redef]
        def transform(self, df):
            return self._transform(df)


class MissingMonaiFlowDependency(ImportError):
    """Raised when optional MONAI-Flow runtime dependencies are not installed."""


def _require_monai_flow():
    try:
        return importlib.import_module("monai_flow")
    except ImportError as e:
        raise MissingMonaiFlowDependency(
            "MONAI-Flow is an optional Pixels integration. Install it in the "
            "Databricks notebook or cluster before logging MONAI bundles."
        ) from e


def log_monai_flow_bundle(*args, **kwargs):
    """Log a MONAI bundle through the optional MONAI-Flow package.

    This wrapper keeps MONAI/MONAI-Flow out of the base Pixels install while
    giving notebooks a stable Pixels import path.
    """
    monai_flow = _require_monai_flow()
    return monai_flow.log_monai_bundle(*args, **kwargs)


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
    """Run a logged MONAI-Flow MLflow pyfunc model against image paths.

    The transformer expects an input column containing paths accessible from the
    Databricks runtime, typically Unity Catalog Volume paths. It adds a struct
    column with `output_dir`, `output_files`, `labels`, and `error`.
    """

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
