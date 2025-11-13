from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from .base import Agent, AgentContext, AgentMessage
from ..services.artifacts import ArtifactStore
from ..services.datasets import DatasetManifest, load_amlb_dataset


DEFAULTS = {
    "random_seed": 42,
    "train_size": 0.8,
    "validation_thresholds": {
        "col_missing": 0.6,
        "high_cardinality_ratio": 0.5,
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _infer_task_type(y: pd.Series) -> Literal["classification", "regression"]:
    if y.dtype == "bool":
        return "classification"
    if pd.api.types.is_numeric_dtype(y):
        nunique = int(y.nunique(dropna=True))
        # small discrete integer targets are likely classification
        if np.allclose(y, y.astype(int), equal_nan=True) and nunique <= 20:
            return "classification"
        return "regression"
    return "classification"


def _detect_temporal_columns(df: pd.DataFrame) -> list[str]:
    temporal_cols: list[str] = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            temporal_cols.append(col)
        else:
            low = col.lower()
            if any(k in low for k in ("date", "time", "timestamp", "ts")):
                # try parse few values to confirm
                sample = df[col].dropna().astype(str).head(20).tolist()
                parsed_ok = 0
                for v in sample:
                    try:
                        pd.to_datetime(v)
                        parsed_ok += 1
                    except Exception:
                        pass
                if parsed_ok >= max(3, len(sample) // 2):
                    temporal_cols.append(col)
    return temporal_cols


def _is_textual_series(s: pd.Series) -> bool:
    if not pd.api.types.is_object_dtype(s) and not pd.api.types.is_string_dtype(s):
        return False
    non_null = s.dropna().astype(str)
    if non_null.empty:
        return False
    avg_len = non_null.map(len).mean()
    unique_ratio = non_null.nunique() / max(1, len(non_null))
    return avg_len > 30 and unique_ratio > 0.3


def _column_entropy(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return 0.0
    if pd.api.types.is_numeric_dtype(s):
        # discretize
        try:
            kb = KBinsDiscretizer(n_bins=16, encode="ordinal", strategy="quantile")
            vals = s.to_numpy().reshape(-1, 1)
            binned = kb.fit_transform(vals).astype(int).ravel()
            counts = np.bincount(binned)
        except Exception:
            counts = np.ones(1, dtype=int)
    else:
        counts = s.value_counts().to_numpy()
    probs = counts / counts.sum()
    # avoid log(0)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


@dataclass(slots=True)
class DataAgentConfig:
    random_seed: int = DEFAULTS["random_seed"]
    train_size: float = DEFAULTS["train_size"]
    col_missing_threshold: float = DEFAULTS["validation_thresholds"]["col_missing"]
    high_cardinality_ratio: float = DEFAULTS["validation_thresholds"]["high_cardinality_ratio"]


class DataAgent(Agent):
    def __init__(self, name: str = "AetherML", context: AgentContext | None = None) -> None:
        super().__init__(name=name, context=context)
        self.cfg = DataAgentConfig()
        self.artifacts = ArtifactStore()

    async def handle(self, message: AgentMessage) -> AgentMessage:
        dataset_id = cast(str, message.payload.get("dataset_id"))
        target_column = cast(str | None, message.payload.get("target"))
        if not dataset_id:
            raise ValueError("dataset_id is required")

        # 1) Load dataset
        manifest = self._load_dataset(dataset_id, target_column=target_column)
        df = pd.read_csv(manifest.local_path)
        y = df[manifest.target_column]
        X = df.drop(columns=[manifest.target_column])
        task_type = _infer_task_type(y)

        # Create run folder
        run_loc = self.artifacts.create_run_location("aetherml", dataset_id)

        # 2) Validate
        validation_report = self._validate_dataset(df, manifest, task_type)
        validation_report_url = self.artifacts.save_json(run_loc, "validation_report.json", validation_report)

        # 3) Split
        split_meta, train_df, test_df = self._split_dataset(X, y, task_type, validation_report)
        train_url = self.artifacts.save_dataframe_csv(run_loc, "splits/train.csv", train_df)
        test_url = self.artifacts.save_dataframe_csv(run_loc, "splits/test.csv", test_df)
        split_meta["train_path"] = train_url
        split_meta["test_path"] = test_url
        split_metadata_url = self.artifacts.save_json(run_loc, "split_metadata.json", split_meta)

        # 4) Metafeatures
        metafeatures = self._compute_metafeatures(df, manifest, task_type, validation_report)
        metafeatures_url = self.artifacts.save_json(run_loc, "metafeatures.json", metafeatures)

        # 5) Register run (MLflow local file store)
        run_metadata = self._register_run(dataset_id, task_type, run_loc.root)
        run_metadata_url = self.artifacts.save_json(run_loc, "run_metadata.json", run_metadata)

        # 6) Recommendations
        recommendation = self._generate_recommendations(validation_report, metafeatures, task_type)
        recommendation_url = self.artifacts.save_json(
            run_loc, "code_agent_recommendation.json", recommendation
        )

        warnings = []
        if validation_report["dataset"]["duplicates"] > 0:
            warnings.append("duplicates_present")
        warnings.extend([f"high_missingness: {c}" for c in validation_report["warnings"].get("high_missing_cols", [])])
        if imbalance := validation_report["warnings"].get("imbalance"):
            warnings.append(f"imbalance: {imbalance}")

        response_payload: dict[str, Any] = {
            "ok": True,
            "dataset_id": dataset_id,
            "manifest": {
                "source_url": manifest.source_url,
                "shape": list(manifest.shape),
                "target": manifest.target_column,
            },
            "validation_report_url": validation_report_url,
            "split_metadata_url": split_metadata_url,
            "metafeatures_url": metafeatures_url,
            "run_metadata_url": run_metadata_url,
            "code_agent_recommendation_url": recommendation_url,
            "warnings": warnings,
            "errors": [],
            "timestamp": _now_iso(),
            "code_agent_recommendation": recommendation,
        }
        return AgentMessage(
            sender=self.name,
            recipient=message.sender,
            content="data_agent_result",
            payload=response_payload,
        )

    # Step 1
    def _load_dataset(self, dataset_id: str, *, target_column: str | None) -> DatasetManifest:
        return load_amlb_dataset(dataset_id, target=target_column)

    # Step 2
    def _validate_dataset(
        self,
        df: pd.DataFrame,
        manifest: DatasetManifest,
        task_type: Literal["classification", "regression"],
    ) -> dict[str, Any]:
        target = manifest.target_column
        X = df.drop(columns=[target])
        y = df[target]

        # types
        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        categorical_cols = [c for c in X.columns if pd.api.types.is_categorical_dtype(X[c]) or pd.api.types.is_object_dtype(X[c])]
        datetime_cols = _detect_temporal_columns(X)
        text_cols = [c for c in categorical_cols if _is_textual_series(X[c])]
        categorical_cols = [c for c in categorical_cols if c not in text_cols]

        # missingness
        missing_per_col = (X.isna().sum() / len(X)).to_dict()
        high_missing_cols = [c for c, r in missing_per_col.items() if r >= self.cfg.col_missing_threshold]

        # high cardinality
        high_card_cols: list[str] = []
        for c in categorical_cols:
            nunique = int(X[c].nunique(dropna=True))
            if nunique / max(1, len(X)) >= self.cfg.high_cardinality_ratio:
                high_card_cols.append(c)

        # duplicates
        duplicates = int(df.duplicated().sum())

        # imbalance/outliers
        imbalance: dict[str, Any] | None = None
        outliers: dict[str, Any] | None = None
        if task_type == "classification":
            vc = y.value_counts(normalize=True)
            min_frac = float(vc.min())
            if min_frac < 0.05:
                imbalance = {"minor_class_frac": min_frac}
        else:
            if y.std(ddof=0) > 0:
                z = np.abs((y - y.mean()) / y.std(ddof=0))
                frac = float((z > 3).mean())
                if frac > 0.01:
                    outliers = {"zscore_gt3_frac": frac}

        # leakage candidates
        leakage_cols: list[str] = []
        if pd.api.types.is_numeric_dtype(y):
            corrs = {}
            for c in numeric_cols:
                x = X[c]
                if x.nunique(dropna=True) < 2:
                    continue
                try:
                    corr = float(pd.concat([x, y], axis=1).corr().iloc[0, 1])
                except Exception:
                    continue
                corrs[c] = corr
                if abs(corr) >= 0.98:
                    leakage_cols.append(c)
        else:
            # use mutual information to flag excessive dependence
            try:
                X_cat = pd.get_dummies(X[categorical_cols], drop_first=True) if categorical_cols else pd.DataFrame(index=X.index)
                scores = mutual_info_classif(X_cat.fillna("NaN"), y.astype(str), discrete_features=True) if not X_cat.empty else np.array([])
                # Not storing per-col MI; we only flag that some MI is extreme.
                if scores.size and float(np.max(scores)) > 1.5:
                    leakage_cols.append("__mi_flag__")
            except Exception:
                pass

        temporal_cols = _detect_temporal_columns(df)

        feature_types = {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "datetime": temporal_cols,
            "text": text_cols,
        }

        report = {
            "dataset": {
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "target": target,
                "task_type": task_type,
                "duplicates": duplicates,
            },
            "feature_types": feature_types,
            "missing_per_column": missing_per_col,
            "high_cardinality": high_card_cols,
            "temporal_columns": temporal_cols,
            "leakage_suspects": leakage_cols,
            "warnings": {
                "high_missing_cols": high_missing_cols,
                "imbalance": imbalance,
                "outliers": outliers,
            },
            "created_at": _now_iso(),
            "source": manifest.to_dict(),
        }
        return report

    # Step 3
    def _split_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: Literal["classification", "regression"],
        validation_report: dict[str, Any],
    ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
        train_size = self.cfg.train_size
        seed = self.cfg.random_seed
        temporal_cols = validation_report.get("temporal_columns", [])
        if task_type == "classification":
            sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
            idx_train, idx_test = next(sss.split(X, y))
            train_df = pd.concat([X.iloc[idx_train], y.iloc[idx_train]], axis=1)
            test_df = pd.concat([X.iloc[idx_test], y.iloc[idx_test]], axis=1)
            method = "stratified"
        elif temporal_cols:
            # temporal split by the first temporal column in original df if present in X
            temporal_col = temporal_cols[0] if temporal_cols else None
            if temporal_col and temporal_col in X.columns:
                order = X[temporal_col].copy()
                try:
                    order = pd.to_datetime(order)
                except Exception:
                    # fallback: lexical order
                    order = order.astype(str)
                sort_idx = order.sort_values().index
            else:
                sort_idx = X.index
            cutoff = int(len(sort_idx) * train_size)
            idx_train = sort_idx[:cutoff]
            idx_test = sort_idx[cutoff:]
            train_df = pd.concat([X.loc[idx_train], y.loc[idx_train]], axis=1)
            test_df = pd.concat([X.loc[idx_test], y.loc[idx_test]], axis=1)
            method = "temporal"
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, random_state=seed
            )
            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)
            method = "random"

        meta = {
            "method": method,
            "seed": seed,
            "train_size": train_size,
            "target": y.name,
            "n_train": int(train_df.shape[0]),
            "n_test": int(test_df.shape[0]),
        }
        return meta, train_df, test_df

    # Step 4
    def _compute_metafeatures(
        self,
        df: pd.DataFrame,
        manifest: DatasetManifest,
        task_type: Literal["classification", "regression"],
        validation_report: dict[str, Any],
    ) -> dict[str, Any]:
        target = manifest.target_column
        y = df[target]
        X = df.drop(columns=[target])
        n_rows, n_cols = df.shape
        missing_rate = float(df.isna().sum().sum() / (n_rows * n_cols))

        # class balance or y stats
        class_balance: dict[str, float] | None = None
        y_stats: dict[str, float] | None = None
        if task_type == "classification":
            vc = y.value_counts(normalize=True)
            class_balance = {str(k): float(v) for k, v in vc.items()}
        else:
            y_stats = {
                "mean": float(np.nanmean(y)),
                "std": float(np.nanstd(y)),
                "skewness": float(pd.Series(y).skew(skipna=True)),
                "kurtosis": float(pd.Series(y).kurt(skipna=True)),
            }

        # correlations (numeric only)
        corr_summary: dict[str, Any] = {}
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if len(num_cols) >= 2:
            corr = X[num_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_pairs = []
            for i, c in enumerate(num_cols):
                for j in range(i + 1, len(num_cols)):
                    c2 = num_cols[j]
                    v = float(upper.loc[c, c2])
                    if v >= 0.95:
                        high_pairs.append((c, c2, v))
            corr_summary = {
                "num_numeric": len(num_cols),
                "high_corr_pairs": [{"c1": a, "c2": b, "corr_abs": float(v)} for a, b, v in high_pairs],
            }

        # feature-level stats
        features: dict[str, Any] = {}
        for col in X.columns:
            s = X[col]
            dtype = str(s.dtype)
            n_unique = int(s.nunique(dropna=True))
            missing = float(s.isna().mean())
            skew = float(pd.Series(s).skew(skipna=True)) if pd.api.types.is_numeric_dtype(s) else np.nan
            kurt = float(pd.Series(s).kurt(skipna=True)) if pd.api.types.is_numeric_dtype(s) else np.nan
            entropy = _column_entropy(s)
            features[col] = {
                "dtype": dtype,
                "n_unique": n_unique,
                "missing_rate": missing,
                "entropy": float(entropy),
                "skewness": float(skew) if not np.isnan(skew) else None,
                "kurtosis": float(kurt) if not np.isnan(kurt) else None,
            }

        return {
            "dataset": {
                "id": manifest.dataset_id,
                "n_rows": int(n_rows),
                "n_cols": int(n_cols),
                "feature_types": validation_report["feature_types"],
                "missing_rate": missing_rate,
                "class_balance": class_balance,
                "y_stats": y_stats,
                "correlations": corr_summary,
            },
            "features": features,
            "created_at": _now_iso(),
        }

    # Step 5
    def _register_run(self, dataset_id: str, task_type: str, run_dir: Path) -> dict[str, Any]:
        # MLflow local run (file store). Tolerate missing mlflow.
        try:
            import mlflow  # type: ignore

            with mlflow.start_run(run_name=f"aetherml:{dataset_id}") as run:
                mlflow.log_params(
                    {
                        "dataset_id": dataset_id,
                        "task_type": task_type,
                    }
                )
                mlflow.log_artifacts(str(run_dir))
                return {
                    "run_id": run.info.run_id,
                    "artifact_uri": run.info.artifact_uri,
                    "start_time": _now_iso(),
                }
        except Exception as e:
            return {
                "run_id": None,
                "artifact_uri": None,
                "start_time": _now_iso(),
                "note": f"mlflow_disabled: {e.__class__.__name__}",
            }

    # Step 6
    def _generate_recommendations(
        self,
        validation_report: dict[str, Any],
        metafeatures: dict[str, Any],
        task_type: Literal["classification", "regression"],
    ) -> dict[str, Any]:
        types = validation_report["feature_types"]
        high_missing = validation_report["warnings"].get("high_missing_cols") or []
        high_card = validation_report.get("high_cardinality") or []

        summary = (
            "Build sklearn pipeline: median impute, one-hot low-cardinality, "
            "target-encode high-cardinality, scale numeric; optional interactions."
        )
        priority = "must"
        steps = [
            {
                "id": "s1_detect",
                "action": "detect_types",
                "tool": "pandas",
                "params": {"infer_datetime": True},
                "example_code": "df.dtypes.to_dict()",
            },
            {
                "id": "s2_impute_numeric",
                "action": "impute",
                "tool": "sklearn",
                "params": {"col_selector": "numeric", "strategy": "median"},
                "example_code": "SimpleImputer(strategy='median')",
            },
            {
                "id": "s3_encode",
                "action": "encode",
                "tool": "category_encoders",
                "params": {
                    "col_selector": "categorical",
                    "method_by_cardinality": {"low": "onehot", "high": "target_encoder"},
                },
                "example_code": "TargetEncoder(cols=high_card_cols)",
            },
            {
                "id": "s4_scale",
                "action": "scale",
                "tool": "sklearn",
                "params": {"col_selector": "numeric", "method": "StandardScaler"},
                "example_code": "StandardScaler()",
            },
            {
                "id": "s5_featuretools",
                "action": "feature_engineer",
                "tool": "featuretools",
                "params": {"primitive_list": ["mean", "std", "count"], "max_depth": 2},
                "example_code": "ft.dfs(entityset=es, target_entity='X')",
            },
        ]

        example_pipeline = (
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n"
            "from category_encoders import TargetEncoder\n"
            "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n"
            "numeric_cols = " + json.dumps(types.get("numeric", [])) + "\n"
            "categorical_cols = " + json.dumps(types.get("categorical", [])) + "\n"
            "high_card_cols = " + json.dumps(high_card) + "\n"
            "low_card_cols = [c for c in categorical_cols if c not in high_card_cols]\n"
            "num_pipe = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])\n"
            "low_cat_pipe = Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])\n"
            "high_cat_pipe = Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', TargetEncoder(cols=high_card_cols))])\n"
            "preproc = ColumnTransformer([\n"
            "    ('num', num_pipe, numeric_cols),\n"
            "    ('low_cat', low_cat_pipe, low_card_cols),\n"
            "    ('high_cat', high_cat_pipe, high_card_cols),\n"
            "])\n"
            f"est = {'RandomForestClassifier()' if task_type=='classification' else 'RandomForestRegressor()'}\n"
            "full_pipeline = Pipeline([('preproc', preproc), ('est', est)])"
        )

        rationale = (
            "Median imputation is robust; target encoding controls dimensionality for high-cardinality "
            "categoricals; scaling stabilizes numeric features; one-hot for low-cardinality. "
            "Consider time-aware splits if temporal columns exist."
        )
        # crude confidence heuristics
        confidence = 0.87
        if high_missing:
            confidence -= 0.05
        if validation_report.get("leakage_suspects"):
            confidence -= 0.05
        confidence = max(0.5, min(confidence, 0.95))

        return {
            "summary": summary,
            "priority": priority,
            "task_type": task_type,
            "steps": steps,
            "example_pipeline_snippet": example_pipeline,
            "frameworks_recommended": ["pandas", "scikit-learn", "category_encoders", "featuretools", "mlflow"],
            "rationale": rationale,
            "estimated_complexity": "medium",
            "confidence": round(confidence, 2),
        }


