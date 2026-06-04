"""正誤判定評価用スクリプト"""

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(pred_csv, gt_csv):
    """
    予測結果と真のラベルを比較して評価指標を計算する。

    Args:
        pred_csv (str): 予測結果のCSVファイルパス
        gt_csv (str): 真のラベルのCSVファイルパス
    """
    # 1. CSV読み込み
    df_pred = pd.read_csv(pred_csv)
    df_gt = pd.read_csv(gt_csv)

    # 2. キーを統一して結合
    # 予測CSVの 'Filename' と 評価用CSVの 'eval_filename' をキーにする
    df = pd.merge(df_pred, df_gt, left_on="Filename", right_on="eval_filename")

    # 3. 数値化 (NORMAL=0, ANOMALY=1)
    # 予測CSV: 'Result' 列
    df["Pred_Label"] = df["Result"].apply(lambda x: 1 if x == "ANOMALY" else 0)

    # 評価用CSV: 'condition' 列
    df["True_Label"] = df["condition"].apply(
        lambda x: 1 if x.lower() == "anomaly" else 0
    )

    # 4. 指標計算
    print("=== 全体評価指標 ===")
    print(
        f"正解率 (Accuracy): {accuracy_score(df['True_Label'], df['Pred_Label']):.4f}"
    )
    print("\n--- 分類レポート ---")
    print(
        classification_report(
            df["True_Label"], df["Pred_Label"], target_names=["NORMAL", "ANOMALY"]
        )
    )

    print("\n=== 混同行列 ===")
    print(confusion_matrix(df["True_Label"], df["Pred_Label"]))

    # 機種ごとの詳細が見たい場合
    print("\n=== 機種ごとの正解率 ===")
    print(
        df.groupby("MachineID").apply(
            lambda x: accuracy_score(x["True_Label"], x["Pred_Label"]),
            include_groups=False,
        )
    )


if __name__ == "__main__":
    evaluate_model(
        "results/supcon_run_XXXXXXXX_XXXXXX/evaluation_results_routed.csv",
        "answer/eval_mapping.csv",
    )
