import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------- #
#       обучение модели         #
# ----------------------------- #

def train_model(X, y, feature_names):
    n = len(y)
    if n < 3:
        raise SystemExit(f"слишком мало размеченных примеров для обучения: {n} (нужно ≥3)")

    cv_folds = max(2, min(5, n))
    print(f"[INFO] размер обучающей выборки: {n}, cv={cv_folds}")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])

    params = {"ridge__alpha": [0.1, 1.0, 5.0, 10.0]}
    grid = GridSearchCV(pipe, params, scoring="neg_mean_absolute_error", cv=cv_folds, n_jobs=-1)
    grid.fit(X, y)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    print(f"[INFO] train MAE: {mae:.2f}° на n={n}")
    return best_model


# ----------------------------- #
#     применение коррекции      #
# ----------------------------- #

def apply_correction(model, df_auto, feature_names):
    # убеждаемся, что все признаки присутствуют, при отсутствии — создаём пустые
    for f in feature_names:
        if f not in df_auto.columns:
            df_auto[f] = np.nan

    X_all = df_auto[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy()
    preds = model.predict(X_all)
    df_auto["angle_corrected"] = preds
    return df_auto


# ----------------------------- #
#         визуализация          #
# ----------------------------- #

def plot_before_after(df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x="true_angle_deg", y="angle_deg", data=df, label="до коррекции", alpha=0.7)
    sns.scatterplot(x="true_angle_deg", y="angle_corrected", data=df, label="после коррекции", alpha=0.7)
    plt.plot([0, df["true_angle_deg"].max()], [0, df["true_angle_deg"].max()], "k--", lw=1)
    plt.xlabel("Истинный угол, °")
    plt.ylabel("Предсказанный угол, °")
    plt.title("До / после коррекции")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "before_after.png", dpi=150)
    plt.close()
    print(f"[OK] график сохранён: {out_dir/'before_after.png'}")


# ----------------------------- #
#            MAIN               #
# ----------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", required=True, help="CSV с автоматическими измерениями")
    parser.add_argument("--labels", required=True, help="CSV с ручными метками")
    parser.add_argument("--out-corrected", required=True, help="Куда сохранить скорректированный CSV")
    parser.add_argument("--reports", default="reports_ml", help="Папка для отчётов и графиков")
    parser.add_argument("--features", type=str, default=None, help="Список признаков через запятую (angle_deg,error,...)")
    args = parser.parse_args()

    df_auto = pd.read_csv(args.auto)
    df_labels = pd.read_csv(args.labels)

    # приводим file-имена к виду без путей
    df_auto["file"] = df_auto["file"].apply(lambda p: Path(str(p)).name)
    df_labels["file"] = df_labels["file"].apply(lambda p: Path(str(p)).name)

    df_merged = df_auto.merge(df_labels[["file", "true_angle_deg"]], on="file", how="inner")
    print(f"[INFO] Совпавших размеченных файлов: {len(df_merged)}")

    # выбираем признаки
    candidate_features = ["angle_deg", "error", "notes"]
    if args.features:
        requested = [s.strip() for s in args.features.split(",") if s.strip()]
    else:
        requested = candidate_features

    feature_names = [f for f in requested if f in df_auto.columns]
    if not feature_names:
        # если нет — создаём angle_deg как fallback
        if "angle_deg" in df_auto.columns:
            feature_names = ["angle_deg"]
        else:
            raise SystemExit("Не найдено ни одного признака для обучения.")

    print(f"[INFO] используем признаки: {feature_names}")

    # обучаем модель
    X = df_merged[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy()
    y = df_merged["true_angle_deg"].to_numpy()

    model = train_model(X, y, feature_names)
    joblib.dump(model, "cobb_correction.pkl")
    print("[OK] saved model: cobb_correction.pkl")

    # применяем к полным данным
    df_corrected = apply_correction(model, df_auto, feature_names)
    df_corrected.to_csv(args.out_corrected, index=False)
    print(f"[OK] сохранён скорректированный CSV: {args.out_corrected}")

    # если есть размеченные — строим график
    if len(df_merged) >= 3:
        df_plot = df_corrected.merge(df_labels[["file", "true_angle_deg"]], on="file", how="left")
        plot_before_after(df_plot.dropna(subset=["true_angle_deg"]), args.reports)


if __name__ == "__main__":
    main()
