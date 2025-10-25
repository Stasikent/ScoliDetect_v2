# калибровка автоматических измерений Cobb угла с помощью данных ручной разметки:
# Сопоставляет автоматические измерения (results.csv) с ручной разметкой (labels.csv)
# Обучает простую линейную модель коррекции (RidgeCV), которая предсказывает ошибку автомата.
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error

                                                                                                                                # обучение модели коррекции Cobb угла по ручным меткам

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", required=True, help="CSV с автоматическими измерениями (results.csv)")
    parser.add_argument("--labels", required=True, help="CSV с ручными метками (labels.csv)")
    parser.add_argument("--out-corrected", required=True, help="куда сохранить скорректированный CSV")
    parser.add_argument("--reports", default="reports_ml", help="папка для графиков")
    args = parser.parse_args()

                                                                                                                                # загрузка данных
    df_auto = pd.read_csv(args.auto)
    df_labels = pd.read_csv(args.labels)

                                                                                                                                # нормализуем имена файлов
    df_auto["file"] = df_auto["file"].astype(str).apply(lambda p: p.split("\\")[-1].split("/")[-1])
    df_labels["file"] = df_labels["file"].astype(str).apply(lambda p: p.split("\\")[-1].split("/")[-1])

    df = df_auto.merge(df_labels[["file", "true_angle_deg"]], on="file", how="left")
    df_labeled = df.dropna(subset=["true_angle_deg"])

    print(f"[INFO] Совпавших размеченных файлов: {len(df_labeled)}")

                                                                                                                                # признаки и цель
    feature_names = ["angle_deg"]                                                                                               # можно добавить ещё признаки позже
    X_df = df_labeled[feature_names].apply(pd.to_numeric, errors="coerce")
    y = df_labeled["true_angle_deg"].astype(float) - df_labeled["angle_deg"].astype(float)

    print(f"[INFO] используем признаки: {feature_names}")
    print(f"[INFO] размер обучающей выборки: {len(X_df)}, cv=5")

                                                                                                                                # пайплайн
    preproc = ColumnTransformer([
        ("num", StandardScaler(), feature_names)
    ])

    ridge = RidgeCV(alphas=np.logspace(-2, 2, 9))
    model = Pipeline([
        ("preproc", preproc),
        ("ridge", ridge)
    ])

                                                                                                                                # обучение
    model.fit(X_df, y)
    best = model
    best_alpha = best.named_steps["ridge"].alpha_
    coef = best.named_steps["ridge"].coef_
    intercept = best.named_steps["ridge"].intercept_

    print(f"[INFO] best alpha: {best_alpha:.4f}")
    print(f"[INFO] coef={coef}, intercept={intercept:.3f}")

    delta_pred_train = best.predict(X_df)
    mae_train = mean_absolute_error(y, delta_pred_train)
    print(f"[INFO] train MAE: {mae_train:.2f}° на n={len(y)}")

                                                                                                                                # сохранить модель
    joblib.dump(best, "cobb_correction.pkl")
    print("[OK] saved model: cobb_correction.pkl")

                                                                                                                                # применить ко всем результатам
    X_all_df = df_auto[feature_names].apply(pd.to_numeric, errors="coerce")
    delta_pred_all = best.predict(X_all_df)
    angle_corr = df_auto["angle_deg"].astype(float) + delta_pred_all

    df_out = df_auto.copy()
    df_out["angle_corrected"] = angle_corr
    df_out.to_csv(args.out_corrected, index=False)
    print(f"[OK] сохранён скорректированный CSV: {args.out_corrected}")

                                                                                                                                # график до/после
    import os
    os.makedirs(args.reports, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(df_labeled["true_angle_deg"], df_labeled["angle_deg"], label="до коррекции", alpha=0.7)
    plt.scatter(df_labeled["true_angle_deg"], df_labeled["angle_deg"] + delta_pred_train,
                label="после коррекции", alpha=0.7)
    plt.plot([0, 60], [0, 60], "k--", lw=1)
    plt.xlabel("Истинный угол (ручная разметка)")
    plt.ylabel("Предсказанный угол")
    plt.title("До и после коррекции")
    plt.legend()
    plt.grid(True)
    plot_path = f"{args.reports}/before_after.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[OK] график сохранён: {plot_path}")

                                                                                                                                # показать первые строки
    print("\n=== Показать первые строки results_corrected.csv ===")
    print(df_out.head())
    print(f"\n[INFO] Всего строк: {len(df_out)}")

if __name__ == "__main__":
    main()
