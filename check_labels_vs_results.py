# check_labels_vs_results.py
# сверяет автоматические измерения (results.csv) с ручной разметкой (labels.csv), показывает сколько файлов размечено, сколько нет, и печатает список пропущенных (у которых нет true_angle_deg).
import pandas as pd
from pathlib import Path
import sys

def main():                                                                                                 # Определение путей
    base = Path(__file__).resolve().parent
    res = base / "results.csv"
    lab = base / "labels.csv"
                                                                                                            # оба файла лежат в той же папке, что и сам .py
    if not res.exists():                                                                                    # Если не найден results.csv или labels.csv — печатает понятную ошибку и делает sys.exit(1)
        print("[ERROR] results.csv не найден в текущей папке.")
        sys.exit(1)
    if not lab.exists():
        print("[ERROR] labels.csv не найден в текущей папке.")
        sys.exit(1)

    auto = pd.read_csv(res, encoding="utf-8")                                                               # это таблица с автоизмерениями
    labs = pd.read_csv(lab, encoding="utf-8")                                                               # таблица ручной разметки

    auto["file"] = auto["file"].astype(str).apply(lambda p: Path(p).name)
    labs["file"] = labs["file"].astype(str).apply(lambda p: Path(p).name)
    labs = labs.drop_duplicates(subset=["file"], keep="last")                                               # Убираем дубликаты разметки
                                                                                                            # Если один файл размечали несколько раз — берётся последняя строка.
    df = auto.merge(labs[["file","true_angle_deg"]], on="file", how="left")                                 # Сопоставление и выявление пропусков
    miss = df[df["true_angle_deg"].isna()][["file","angle_deg"]].sort_values("file")

    print("="*50)
    print("Сверка results.csv ↔ labels.csv")
    print("="*50)
    print(f"[INFO] Всего файлов в results.csv: {len(auto)}")
    print(f"[INFO] Размеченных (найдено в labels): {len(df) - len(miss)}")
    print(f"[INFO] Без метки: {len(miss)}")

    if len(miss):
        print("\n[СПИСОК без метки (file, auto_angle)]:")
        with pd.option_context("display.max_rows", None, "display.max_colwidth", 200):
            print(miss.to_string(index=False))
    else:
        print("\n[OK] Все файлы из results.csv нашли соответствие в labels.csv")

if __name__ == "__main__":
    main()
