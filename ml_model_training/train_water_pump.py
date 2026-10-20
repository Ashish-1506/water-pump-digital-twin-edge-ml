from __future__ import annotations



import json

import re

import warnings

from pathlib import Path

from typing import Any



import joblib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import tensorflow as tf



from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (

    ConfusionMatrixDisplay,

    accuracy_score,

    average_precision_score,

    classification_report,

    confusion_matrix,

    f1_score,

    precision_recall_curve,

    precision_recall_fscore_support,

    roc_auc_score,

    roc_curve,

    auc,

)

from sklearn.preprocessing import StandardScaler, label_binarize

from sklearn.inspection import permutation_importance





# ============================== CONFIG ==============================



DATA_DIR = Path("dataset")

OUTPUT_DIR = Path("model_output")

PLOT_DIR = OUTPUT_DIR / "report_figures"



FEATURES = [

    "vib_rms",

    "vib_peak",

    "vib_x",

    "vib_y",

    "vib_z",

    "current",

    "temperature",

]

TARGET = "label"

CLASS_NAMES = [

    "Normal",

    "Dry Run",

    "Overheating",

    "Abnormal Operating Condition",

]

EXPECTED_LABELS = {0, 1, 2, 3}

RANDOM_SEED = 42



# Preserve the session-wise design used in the supplied code.

HOLDOUT_SESSION = "session_1"

DEVELOPMENT_SESSIONS = ["session_2", "session_3"]



# Optional: if timestamps are numeric sample counters, plotting falls back to index.

MAX_SCATTER_POINTS_PER_CLASS = 2500



OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR.mkdir(parents=True, exist_ok=True)



np.random.seed(RANDOM_SEED)

tf.random.set_seed(RANDOM_SEED)



plt.rcParams.update({

    "figure.dpi": 120,

    "savefig.dpi": 300,

    "font.size": 9,

    "axes.titlesize": 11,

    "axes.labelsize": 9,

    "legend.fontsize": 8,

})





# ============================== UTILITIES ==============================



def safe_name(value: str) -> str:

    """Convert a label or filename to a portable artifact name."""

    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_").lower()





def save_current_figure(filename: str) -> None:

    """Save and close the current Matplotlib figure."""

    plt.tight_layout()

    plt.savefig(PLOT_DIR / filename, bbox_inches="tight")

    plt.close()





def get_session_id(filename: str) -> str:

    """Extract session_1/session_2/etc. from a filename."""

    match = re.search(r"(session[_-]?\d+)", filename, flags=re.IGNORECASE)

    if match:

        return match.group(1).lower().replace("-", "_")

    return "session_unknown"





def read_manifest(path: Path) -> dict[str, Any]:

    """Read optional JSON manifest mapping filenames to metadata."""

    if not path.exists():

        warnings.warn(

            f"No {path} found. Provenance may remain UNKNOWN_ORIGIN. "

            "Create a manifest for a research report.",

            RuntimeWarning,

        )

        return {}

    try:

        value = json.loads(path.read_text(encoding="utf-8"))

    except json.JSONDecodeError as exc:

        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(value, dict):

        raise ValueError("dataset_manifest.json must contain a JSON object.")

    return value





def infer_origin(

    filename: str,

    df: pd.DataFrame,

    manifest: dict[str, Any]

) -> str:

    """Determine recording origin from the manifest or CSV metadata."""



    entry = manifest.get(filename)



    if isinstance(entry, str):

        return entry.strip() or "UNKNOWN_ORIGIN"



    if isinstance(entry, dict):

        origin = entry.get("origin")

        if origin:

            return str(origin).strip()



    if "data_origin" in df.columns:

        values = (

            df["data_origin"]

            .dropna()

            .astype(str)

            .str.strip()

            .unique()

        )



        if len(values) == 1 and values[0]:

            return values[0]



        if len(values) > 1:

            return "MIXED_ORIGIN"



    return "UNKNOWN_ORIGIN"





def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray,

                   labels: list[int] | None = None) -> dict[str, Any]:

    """Print metrics and save count and row-normalized confusion matrices."""

    if labels is None:

        labels = sorted(EXPECTED_LABELS)



    target_names = [CLASS_NAMES[i] for i in labels]

    accuracy = accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(

        y_true, y_pred, labels=labels, average="macro", zero_division=0

    )

    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(

        y_true, y_pred, labels=labels, average="weighted", zero_division=0

    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)



    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")

    print(f"Accuracy: {accuracy:.4f}")

    print(classification_report(

        y_true, y_pred, labels=labels, target_names=target_names,

        digits=4, zero_division=0

    ))

    print("Confusion matrix (rows=actual, columns=predicted):")

    print(cm)



    return {

        "accuracy": float(accuracy),

        "macro_precision": float(macro_p),

        "macro_recall": float(macro_r),

        "macro_f1": float(macro_f1),

        "weighted_precision": float(weighted_p),

        "weighted_recall": float(weighted_r),

        "weighted_f1": float(weighted_f1),

        "classification_report": classification_report(

            y_true, y_pred, labels=labels, target_names=target_names,

            output_dict=True, zero_division=0

        ),

        "confusion_matrix": cm.tolist(),

        "labels_in_matrix": labels,

    }





def build_mlp(n_features: int) -> tf.keras.Model:

    """Build the compact four-class MLP."""

    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=(n_features,), name="sensor_features"),

        tf.keras.layers.Dense(32, activation="relu"),

        tf.keras.layers.Dropout(0.15),

        tf.keras.layers.Dense(16, activation="relu"),

        tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax",

                              name="class_probabilities"),

    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"],

    )

    return model





def make_mlp_callbacks() -> list[tf.keras.callbacks.Callback]:

    return [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss", patience=15, restore_best_weights=True

        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5

        ),

    ]





# ============================== LOAD AND VALIDATE ==============================



csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:

    raise FileNotFoundError(

        f"No CSV files found in {DATA_DIR.resolve()}. "

        "Place the sensor recordings in the dataset folder."

    )



manifest = read_manifest(DATA_DIR / "dataset_manifest.json")

frames: list[pd.DataFrame] = []

file_audit: list[dict[str, Any]] = []



for csv_path in csv_files:

    df = pd.read_csv(csv_path)

    required = set(FEATURES + [TARGET])

    missing = required - set(df.columns)

    if missing:

        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")



    df = df.copy()

    for column in FEATURES + [TARGET]:

        df[column] = pd.to_numeric(df[column], errors="coerce")



    if df[FEATURES + [TARGET]].isna().any().any():

        bad_columns = df[FEATURES + [TARGET]].columns[

            df[FEATURES + [TARGET]].isna().any()

        ].tolist()

        raise ValueError(

            f"{csv_path.name} contains missing/non-numeric values in {bad_columns}."

        )



    feature_array = df[FEATURES].to_numpy(dtype=np.float64)

    if not np.isfinite(feature_array).all():

        raise ValueError(f"{csv_path.name} contains infinite feature values.")



    raw_labels = df[TARGET].to_numpy(dtype=np.float64)

    if not np.isfinite(raw_labels).all():

        raise ValueError(f"{csv_path.name} contains invalid labels.")

    if not np.equal(raw_labels, np.floor(raw_labels)).all():

        raise ValueError(f"{csv_path.name} contains non-integer labels.")

    df[TARGET] = raw_labels.astype(np.int64)



    labels_in_file = set(df[TARGET].unique())

    if not labels_in_file.issubset(EXPECTED_LABELS):

        raise ValueError(

            f"{csv_path.name} contains unexpected labels: "

            f"{sorted(labels_in_file - EXPECTED_LABELS)}"

        )



    origin = infer_origin(csv_path.name, df, manifest)

    df["data_origin"] = origin

    df["session_id"] = get_session_id(csv_path.name)

    df["source_file"] = csv_path.name

    if "condition_name" not in df.columns:

        df["condition_name"] = df[TARGET].map(

            {i: CLASS_NAMES[i] for i in range(len(CLASS_NAMES))}

        )



    frames.append(df)

    file_audit.append({

        "file": csv_path.name,

        "rows": int(len(df)),

        "session_id": df["session_id"].iloc[0],

        "origin": origin,

        "labels": sorted(int(x) for x in labels_in_file),

    })



data = pd.concat(frames, ignore_index=True)



present_labels = set(data[TARGET].unique())

if present_labels != EXPECTED_LABELS:

    raise ValueError(

        f"Dataset labels are {sorted(present_labels)}; expected "

        f"{sorted(EXPECTED_LABELS)}. Check the CSV label column."

    )



class3_origins = sorted(

    data.loc[data[TARGET] == 3, "data_origin"]

    .astype(str)

    .str.lower()

    .unique()

)



if "UNKNOWN_ORIGIN" in set(data["data_origin"]):

    warnings.warn(

        "Some recordings have UNKNOWN_ORIGIN. "

        "Verify their origin before reporting experimental results.",

        RuntimeWarning,

    )



# Dataset audit artifacts

data.to_csv(OUTPUT_DIR / "validated_dataset_with_metadata.csv", index=False)

pd.DataFrame(file_audit).to_csv(OUTPUT_DIR / "dataset_file_audit.csv", index=False)



print(f"Loaded {len(data):,} rows from {len(csv_files)} CSV files.")

print("\nRows by class:")

print(data[TARGET].map(dict(enumerate(CLASS_NAMES))).value_counts())

print("\nRows by origin:")

print(data["data_origin"].value_counts(dropna=False))

print("\nRows by session:")

print(data["session_id"].value_counts(dropna=False))





# ============================== SPLIT BY SESSION ==============================



# Exact session ID matching avoids accidentally treating session_10 as session_1.

test_mask = data["session_id"].eq(HOLDOUT_SESSION.lower().replace("-", "_"))

if not test_mask.any():

    raise ValueError(

        f"No recordings contain '{HOLDOUT_SESSION}' in their filename. "

        "Update HOLDOUT_SESSION or rename the session files consistently."

    )



train_df = data.loc[~test_mask].copy()

test_df = data.loc[test_mask].copy()



if train_df.empty or test_df.empty:

    raise ValueError("The session-based train/test split produced an empty partition.")



if set(train_df[TARGET].unique()) != EXPECTED_LABELS:

    raise ValueError(

        "Training partition must contain all four classes. Check session assignments."

    )

if set(test_df[TARGET].unique()) != EXPECTED_LABELS:

    raise ValueError(

        "Held-out test partition must contain all four classes. "

        "If the test session does not contain every class, revise the protocol "

        "and report the limitation instead of fabricating samples."

    )



print("\nTraining files:", sorted(train_df["source_file"].unique()))

print("Held-out test files:", sorted(test_df["source_file"].unique()))



X_train = train_df[FEATURES].to_numpy(dtype=np.float32)

y_train = train_df[TARGET].to_numpy(dtype=np.int64)

X_test = test_df[FEATURES].to_numpy(dtype=np.float32)

y_test = test_df[TARGET].to_numpy(dtype=np.int64)



# No feature multipliers are applied. The same standardization pipeline is

# saved and used consistently for Logistic Regression and the final MLP.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)

X_test_scaled = scaler.transform(X_test).astype(np.float32)





# ============================== SESSION-LEVEL DEVELOPMENT CHECK ==============================



cv_rows: list[dict[str, Any]] = []

session_names = sorted(train_df["session_id"].unique())



if len(session_names) >= 2:

    for val_session in session_names:

        cv_train_df = train_df.loc[train_df["session_id"] != val_session]

        cv_val_df = train_df.loc[train_df["session_id"] == val_session]



        if cv_train_df.empty or cv_val_df.empty:

            continue

        if set(cv_train_df[TARGET].unique()) != EXPECTED_LABELS:

            warnings.warn(

                f"Skipping CV fold for {val_session}: training fold lacks a class."

            )

            continue

        if set(cv_val_df[TARGET].unique()) != EXPECTED_LABELS:

            warnings.warn(

                f"CV validation session {val_session} does not contain every class; "

                "metrics may not represent all classes."

            )



        fold_scaler = StandardScaler()

        X_fold_train = fold_scaler.fit_transform(

            cv_train_df[FEATURES].to_numpy(dtype=np.float32)

        )

        X_fold_val = fold_scaler.transform(

            cv_val_df[FEATURES].to_numpy(dtype=np.float32)

        )

        y_fold_train = cv_train_df[TARGET].to_numpy(dtype=np.int64)

        y_fold_val = cv_val_df[TARGET].to_numpy(dtype=np.int64)



        fold_rf = RandomForestClassifier(

            n_estimators=300,

            max_depth=None,

            min_samples_leaf=2,

            class_weight="balanced",

            random_state=RANDOM_SEED,

            n_jobs=-1,

        )

        fold_rf.fit(X_fold_train, y_fold_train)

        fold_pred = fold_rf.predict(X_fold_val)



        cv_rows.append({

            "validation_session": val_session,

            "n_train": int(len(cv_train_df)),

            "n_validation": int(len(cv_val_df)),

            "accuracy": float(accuracy_score(y_fold_val, fold_pred)),

            "macro_f1": float(f1_score(

                y_fold_val, fold_pred, labels=[0, 1, 2, 3],

                average="macro", zero_division=0

            )),

        })

else:

    warnings.warn(

        "Fewer than two training sessions: session-level cross-validation skipped."

    )



cv_df = pd.DataFrame(cv_rows)

cv_df.to_csv(OUTPUT_DIR / "session_level_validation.csv", index=False)

if not cv_df.empty:

    print("\nSession-level development results:")

    print(cv_df.to_string(index=False))





# ============================== TRAIN AND EVALUATE CLASSICAL MODELS ==============================



lr = LogisticRegression(

    max_iter=2000,

    class_weight="balanced",

    random_state=RANDOM_SEED,

    solver="lbfgs",

)

lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)

lr_prob = lr.predict_proba(X_test_scaled)

lr_metrics = evaluate_model("Logistic Regression", y_test, lr_pred)



rf = RandomForestClassifier(

    n_estimators=300,

    max_depth=None,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=RANDOM_SEED,

    n_jobs=-1,

)

rf.fit(X_train, y_train)  # Trees do not require feature standardization.

rf_pred = rf.predict(X_test)

rf_prob = rf.predict_proba(X_test)

rf_metrics = evaluate_model("Random Forest", y_test, rf_pred)



# Explicit feature ablation for Random Forest: full features vs without current.

# This is a diagnostic comparison, not feature weighting.

ablation_rows = []

feature_sets = {

    "all_features": FEATURES,

    "without_current": [f for f in FEATURES if f != "current"],

    "vibration_only": ["vib_rms", "vib_peak", "vib_x", "vib_y", "vib_z"],

}

for variant_name, feature_set in feature_sets.items():

    variant_rf = RandomForestClassifier(

        n_estimators=300,

        max_depth=None,

        min_samples_leaf=2,

        class_weight="balanced",

        random_state=RANDOM_SEED,

        n_jobs=-1,

    )

    variant_rf.fit(train_df[feature_set].to_numpy(dtype=np.float32), y_train)

    variant_pred = variant_rf.predict(

        test_df[feature_set].to_numpy(dtype=np.float32)

    )

    ablation_rows.append({

        "variant": variant_name,

        "features": ", ".join(feature_set),

        "accuracy": float(accuracy_score(y_test, variant_pred)),

        "macro_f1": float(f1_score(

            y_test, variant_pred, labels=[0, 1, 2, 3],

            average="macro", zero_division=0

        )),

    })

pd.DataFrame(ablation_rows).to_csv(

    OUTPUT_DIR / "random_forest_feature_ablation.csv", index=False

)





# ============================== MLP: SELECT EPOCHS, REFIT ON ALL TRAIN SESSIONS ==============================



# Use one development session for early stopping when possible, then refit

# a fresh final MLP on all training sessions using the saved final scaler.

mlp_val_session = None

for candidate in reversed(DEVELOPMENT_SESSIONS):

    if candidate in set(train_df["session_id"]):

        mlp_val_session = candidate

        break

if mlp_val_session is None:

    remaining_sessions = sorted(train_df["session_id"].unique())

    if len(remaining_sessions) >= 2:

        mlp_val_session = remaining_sessions[-1]



best_epochs = 100

selection_history = None



if mlp_val_session is not None:

    mlp_dev_train_df = train_df.loc[train_df["session_id"] != mlp_val_session]

    mlp_dev_val_df = train_df.loc[train_df["session_id"] == mlp_val_session]



    if (

        not mlp_dev_train_df.empty

        and not mlp_dev_val_df.empty

        and set(mlp_dev_train_df[TARGET].unique()) == EXPECTED_LABELS

    ):

        selection_scaler = StandardScaler()

        X_dev_train = selection_scaler.fit_transform(

            mlp_dev_train_df[FEATURES].to_numpy(dtype=np.float32)

        ).astype(np.float32)

        X_dev_val = selection_scaler.transform(

            mlp_dev_val_df[FEATURES].to_numpy(dtype=np.float32)

        ).astype(np.float32)

        y_dev_train = mlp_dev_train_df[TARGET].to_numpy(dtype=np.int64)

        y_dev_val = mlp_dev_val_df[TARGET].to_numpy(dtype=np.int64)



        selector = build_mlp(len(FEATURES))

        selection_history = selector.fit(

            X_dev_train, y_dev_train,

            validation_data=(X_dev_val, y_dev_val),

            epochs=200,

            batch_size=32,

            callbacks=make_mlp_callbacks(),

            verbose=1,

        )

        val_losses = selection_history.history.get("val_loss", [])

        if val_losses:

            best_epochs = int(np.argmin(val_losses) + 1)

        print(f"\nMLP selected epoch count: {best_epochs}")

    else:

        warnings.warn(

            "Could not construct the requested session-based MLP validation split. "

            "Using 100 epochs for final refit; report this limitation."

        )

else:

    warnings.warn("MLP epoch selection unavailable; using 100 epochs for final refit.")



# Final model is trained on every training row. The same scaler object saved

# below is the scaler used for final MLP training and inference.

X_train_mlp = scaler.transform(X_train).astype(np.float32)

X_test_mlp = scaler.transform(X_test).astype(np.float32)



mlp = build_mlp(len(FEATURES))

final_history = mlp.fit(

    X_train_mlp,

    y_train,

    epochs=best_epochs,

    batch_size=32,

    shuffle=True,

    verbose=1,

)



mlp_prob = mlp.predict(X_test_mlp, verbose=0)

mlp_pred = np.argmax(mlp_prob, axis=1)

mlp_metrics = evaluate_model("MLP", y_test, mlp_pred)





# ============================== HARDWARE-ONLY METRICS ==============================



# Exclude only class-3 ground-truth rows, but retain all four prediction labels

# so false predictions as class 3 remain visible in the confusion matrix.

hardware_mask = y_test != 3

hardware_metrics: dict[str, Any] = {}



if hardware_mask.any():

    hardware_metrics["logistic_regression"] = evaluate_model(

        "Logistic Regression Hardware-Only",

        y_test[hardware_mask], lr_pred[hardware_mask], labels=[0, 1, 2, 3]

    )

    hardware_metrics["random_forest"] = evaluate_model(

        "Random Forest Hardware-Only",

        y_test[hardware_mask], rf_pred[hardware_mask], labels=[0, 1, 2, 3]

    )

    hardware_metrics["mlp"] = evaluate_model(

        "MLP Hardware-Only",

        y_test[hardware_mask], mlp_pred[hardware_mask], labels=[0, 1, 2, 3]

    )

else:

    warnings.warn("No class 0/1/2 test samples available for hardware-only evaluation.")





# ============================== SAVE MODELS AND PREPROCESSING ==============================



joblib.dump(lr, OUTPUT_DIR / "logistic_regression.pkl")

joblib.dump(rf, OUTPUT_DIR / "random_forest.pkl")

joblib.dump(scaler, OUTPUT_DIR / "standard_scaler.pkl")

mlp.save(OUTPUT_DIR / "pump_condition_mlp.keras")



converter = tf.lite.TFLiteConverter.from_keras_model(mlp)

tflite_bytes = converter.convert()

(OUTPUT_DIR / "pump_condition_mlp.tflite").write_bytes(tflite_bytes)



# Export the exact feature order, scaler and model preprocessing definition.

preprocessing = {

    "features": FEATURES,

    "scaler_mean": scaler.mean_.tolist(),

    "scaler_scale": scaler.scale_.tolist(),

    "feature_multipliers": [1.0] * len(FEATURES),

    "formula": "model_input[i] = (raw_feature[i] - scaler_mean[i]) / scaler_scale[i]",

    "class_names": {str(i): name for i, name in enumerate(CLASS_NAMES)},

    "target_labels": sorted(EXPECTED_LABELS),

    "notes": [

        "Scaler was fitted on all training sessions, excluding the held-out test session.",

        "Random Forest is trained on raw features; Logistic Regression and MLP use standardized features.",

        "TFLite model does not include StandardScaler preprocessing.",

        "Firmware must use the exact feature definitions, units, order, means, and scales.",

    ],

}

(OUTPUT_DIR / "preprocessing_config.json").write_text(

    json.dumps(preprocessing, indent=2), encoding="utf-8"

)

(OUTPUT_DIR / "feature_order.json").write_text(

    json.dumps(FEATURES, indent=2), encoding="utf-8"

)

(OUTPUT_DIR / "class_names.json").write_text(

    json.dumps({i: name for i, name in enumerate(CLASS_NAMES)}, indent=2),

    encoding="utf-8",

)



# TFLite inference parity check: compare exported model to Keras on test samples.

interpreter = tf.lite.Interpreter(

    model_path=str(OUTPUT_DIR / "pump_condition_mlp.tflite")

)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()



tflite_input = input_details[0]

tflite_output = output_details[0]

tflite_predictions = []



for row in X_test_mlp.astype(tflite_input["dtype"]):

    interpreter.set_tensor(tflite_input["index"], row.reshape(1, -1))

    interpreter.invoke()

    tflite_predictions.append(

        interpreter.get_tensor(tflite_output["index"])[0]

    )



tflite_prob = np.asarray(tflite_predictions)

tflite_pred = np.argmax(tflite_prob, axis=1)

tflite_max_abs_diff = float(np.max(np.abs(mlp_prob - tflite_prob)))

tflite_class_agreement = float(np.mean(mlp_pred == tflite_pred))

print(f"\nKeras/TFLite max probability difference: {tflite_max_abs_diff:.8f}")

print(f"Keras/TFLite predicted class agreement: {tflite_class_agreement:.4%}")



# Model metrics

metrics = {

    "evaluation_protocol": {

        "holdout_session": HOLDOUT_SESSION,

        "train_files": sorted(train_df["source_file"].unique().tolist()),

        "test_files": sorted(test_df["source_file"].unique().tolist()),

        "development_validation": cv_rows,

        "class_3_origins": class3_origins,

        "important_limitation": (

            "Performance applies to this dataset and split only. Independent "

            "physical sessions, verified fault labels, and deployment testing "

            "are required before claiming general real-world performance."

        ),

    },

    "logistic_regression": lr_metrics,

    "random_forest": rf_metrics,

    "mlp": mlp_metrics,

    "hardware_only": hardware_metrics,

    "mlp_selected_epochs": best_epochs,

    "tflite_size_bytes": len(tflite_bytes),

    "tflite_max_abs_probability_difference": tflite_max_abs_diff,

    "tflite_class_agreement": tflite_class_agreement,

    "feature_order": FEATURES,

    "class_names": {str(i): name for i, name in enumerate(CLASS_NAMES)},

}

(OUTPUT_DIR / "metrics.json").write_text(

    json.dumps(metrics, indent=2), encoding="utf-8"

)





# ============================== REPORT FIGURES ==============================

# Exactly ten consolidated report figures are generated below.
for old_plot in PLOT_DIR.glob("*.png"):
    old_plot.unlink()


# 1. Class distribution across the training and held-out test sets.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, frame, title in [
    (axes[0], train_df, "Training Set Class Distribution"),
    (axes[1], test_df, "Held-Out Test Set Class Distribution"),
]:
    counts = (
        frame[TARGET].map(dict(enumerate(CLASS_NAMES)))
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
    )
    counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    ax.tick_params(axis="x", rotation=25)
save_current_figure("01_class_distribution_train_test.png")

# 2. Feature distributions by class (all seven features in one figure).
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()
for i, feature in enumerate(FEATURES):
    ax = axes[i]
    for label in sorted(EXPECTED_LABELS):
        vals = data.loc[data[TARGET] == label, feature].to_numpy()
        if vals.size:
            ax.hist(vals, bins=30, alpha=0.4, density=True, label=CLASS_NAMES[label])
    ax.set_title(feature)
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
axes[-1].axis("off")
fig.suptitle("Sensor Feature Distributions by Class")
save_current_figure("02_feature_distributions_by_class.png")

# 3. Sensor feature correlation heatmap.
corr = data[FEATURES].corr()
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=45, ha="right")
ax.set_yticks(range(len(FEATURES)), FEATURES)
ax.set_title("Sensor Feature Correlation Matrix")
for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, label="Pearson correlation")
save_current_figure("03_sensor_correlation_heatmap.png")

# 4. Two useful sensor-pair scatter plots in one figure.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (x_feature, y_feature) in zip(
    axes, [("vib_rms", "vib_peak"), ("current", "temperature")]
):
    for label in sorted(EXPECTED_LABELS):
        subset = data.loc[data[TARGET] == label]
        if len(subset) > MAX_SCATTER_POINTS_PER_CLASS:
            subset = subset.sample(
                MAX_SCATTER_POINTS_PER_CLASS, random_state=RANDOM_SEED
            )
        ax.scatter(
            subset[x_feature], subset[y_feature], s=12, alpha=0.4,
            label=CLASS_NAMES[label]
        )
    ax.set_title(f"{y_feature} vs {x_feature}")
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.legend(fontsize=7)
save_current_figure("04_sensor_pair_scatter_plots.png")

# 5. Representative vibration and current time series for each class.
fig, axes = plt.subplots(4, 2, figsize=(14, 12))
for label in sorted(EXPECTED_LABELS):
    class_data = data.loc[data[TARGET] == label]
    representative = class_data.loc[
        class_data["source_file"] == class_data["source_file"].iloc[0]
    ].head(500)
    for col, feature in enumerate(["vib_rms", "current"]):
        ax = axes[label, col]
        ax.plot(np.arange(len(representative)), representative[feature], linewidth=0.8)
        ax.set_title(f"{CLASS_NAMES[label]} — {feature}")
        ax.set_xlabel("Sample index")
        ax.set_ylabel(feature)
        ax.grid(True, alpha=0.25)
save_current_figure("05_representative_sensor_time_series.png")

# 6. PCA projection; PCA is fitted only on the training data.
pca_scaler = StandardScaler()
X_train_pca = pca_scaler.fit_transform(X_train)
X_all_pca = pca_scaler.transform(data[FEATURES].to_numpy(dtype=np.float32))
pca = PCA(n_components=2, random_state=RANDOM_SEED)
pca.fit(X_train_pca)
X_pca = pca.transform(X_all_pca)

plt.figure(figsize=(8, 6))
data_labels = data[TARGET].to_numpy()
for label in sorted(EXPECTED_LABELS):
    mask = data_labels == label
    plt.scatter(
        X_pca[mask, 0], X_pca[mask, 1], s=12, alpha=0.45,
        label=CLASS_NAMES[label]
    )
plt.title("PCA Projection of Sensor Feature Space")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
plt.legend()
save_current_figure("06_pca_feature_space.png")

# 7. Random Forest permutation feature importance on held-out test data.
perm = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=RANDOM_SEED,
    scoring="f1_macro", n_jobs=-1
)
perm_df = pd.DataFrame({
    "feature": FEATURES,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean")
perm_df.to_csv(OUTPUT_DIR / "random_forest_permutation_importance.csv", index=False)
plt.figure(figsize=(8, 5))
plt.barh(
    perm_df["feature"], perm_df["importance_mean"],
    xerr=perm_df["importance_std"]
)
plt.title("Random Forest Permutation Importance on Held-Out Test Set")
plt.xlabel("Decrease in macro-F1 after permutation")
save_current_figure("07_random_forest_permutation_importance.png")

# 8. Confusion matrices for the three models in one figure.
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (model_name, predictions) in zip(
    axes,
    [
        ("Logistic Regression", lr_pred),
        ("Random Forest", rf_pred),
        ("MLP", mlp_pred),
    ],
):
    cm = confusion_matrix(y_test, predictions, labels=sorted(EXPECTED_LABELS))
    display = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False,
                 xticks_rotation=45)
    ax.set_title(model_name)
save_current_figure("08_model_confusion_matrices.png")

# 9. Model performance comparison.
comparison_rows = []
for model_name, predictions in {
    "Logistic Regression": lr_pred,
    "Random Forest": rf_pred,
    "MLP": mlp_pred,
}.items():
    p, r, f, _ = precision_recall_fscore_support(
        y_test, predictions, labels=[0, 1, 2, 3],
        average="macro", zero_division=0
    )
    comparison_rows.append({
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f,
    })
comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(PLOT_DIR / "model_comparison_metrics.csv", index=False)
ax = comparison_df.set_index("model").plot(
    kind="bar", figsize=(10, 6), ylim=(0, 1.05)
)
ax.set_title("Model Performance Comparison")
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.tick_params(axis="x", rotation=0)
save_current_figure("09_model_performance_comparison.png")

# 10. ROC and precision-recall curves, consolidated into two panels.
y_binary = label_binarize(y_test, classes=[0, 1, 2, 3])
probability_sets = {
    "Logistic Regression": lr_prob,
    "Random Forest": rf_prob,
    "MLP": mlp_prob,
}
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
roc_ax, pr_ax = axes
for model_name, probabilities in probability_sets.items():
    for label in range(len(CLASS_NAMES)):
        if len(np.unique(y_binary[:, label])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_binary[:, label], probabilities[:, label])
        roc_ax.plot(fpr, tpr, label=f"{model_name} — {CLASS_NAMES[label]}")
        precision, recall, _ = precision_recall_curve(
            y_binary[:, label], probabilities[:, label]
        )
        pr_ax.plot(recall, precision, label=f"{model_name} — {CLASS_NAMES[label]}")
roc_ax.plot([0, 1], [0, 1], "--", label="Chance")
roc_ax.set_title("Multiclass One-vs-Rest ROC Curves")
roc_ax.set_xlabel("False Positive Rate")
roc_ax.set_ylabel("True Positive Rate")
roc_ax.legend(fontsize=6, loc="lower right")
pr_ax.set_title("Multiclass Precision-Recall Curves")
pr_ax.set_xlabel("Recall")
pr_ax.set_ylabel("Precision")
pr_ax.legend(fontsize=6, loc="best")
save_current_figure("10_roc_and_precision_recall_curves.png")

plot_summary = {
    "figure_directory": str(PLOT_DIR.resolve()),
    "png_count": len(list(PLOT_DIR.glob("*.png"))),
    "feature_count": len(FEATURES),
    "classes": CLASS_NAMES,
}
(PLOT_DIR / "plot_generation_summary.json").write_text(
    json.dumps(plot_summary, indent=2), encoding="utf-8"
)

print(f"\nAll artifacts saved to: {OUTPUT_DIR.resolve()}")
print(f"Report figures saved to: {PLOT_DIR.resolve()}")
print(f"Generated PNG figures: {plot_summary['png_count']}")
print("Review dataset_file_audit.csv and provenance before final report claims.")
