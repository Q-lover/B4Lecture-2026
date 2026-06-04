"""対照学習を用いた異常検知モデルの訓練と評価を行うスクリプト."""

import csv
import datetime
import json
import os
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from dataloaders.dataloader import create_dataloader
from pytorch_metric_learning import losses
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import WeightedRandomSampler, random_split

# 定数定義
TRAIN_DIR = "data/dev"
EVAL_DIR = "data/eval"
BATCH_SIZE = 32
EPOCHS = 50  # Early Stopping で実際は早めに止まる
EARLY_STOPPING_PATIENCE = 5  # Val Loss が改善しないエポック数の上限
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EMBEDDING_DIM = 128
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # .to(DEVICE)でモデルやデータを GPU に移動可能
MACHINES = [
    "model_03",
    "model_04",
    "model_05",
    "model_06",
]


def set_seed(seed: int = 42) -> None:
    """乱数シードを設定.

    Args:
        seed (int): シード値
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(42)  # 再現性のためにシード値を固定


# ResNet18 をベースにした対照学習・分類モデル
class SupConClassifier(nn.Module):
    """ResNet18 ベースのエンコーダ + Projection head + Classification head.

    学習時は SupConLoss（対照学習）と BCEWithLogitsLoss（二値分類）の
    両方を使って重みを更新する。
    推論時は classification_head の出力に sigmoid を適用した
    異常確率（0〜1）をスコアとして使用する。
    """

    def __init__(self, out_dim: int = 128) -> None:
        """ResNet18 をベースにした対照学習・分類モデルの初期化.

        Args:
            out_dim (int): Projection head の出力次元数（埋め込みベクトルの次元）
        """
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 最終全結合層を除いた畳み込み部分をエンコーダとして使用
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features  # 512

        # 対照学習用：512 → 512 → 128 次元に変換し L2 正規化
        # Dropout(0.3) で過学習を抑制
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(self.feature_dim, out_dim),
        )
        # 分類用：128 → 1 次元（学習・推論の両方で使用）
        self.classification_head = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """入力画像をエンコードして埋め込みと分類スコアを出力.

        Args:
            x (torch.Tensor): 入力画像

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 埋め込みと分類スコア
        """
        features = torch.flatten(self.encoder(x), 1)
        embeddings = nn.functional.normalize(
            self.projection_head(features), dim=1
        )  # L2 正規化された埋め込みベクトル
        outputs = self.classification_head(
            embeddings
        )  # BCEWithLogitsLoss で使用するスコア（sigmoid を適用する前の値）
        return embeddings, outputs


def count_labels(file_list: list[str]) -> dict:
    """ファイルリスト内の normal / anomaly 件数を返す.

    Args:
        file_list (list[str]): 音声ファイルのパスのリスト

    Returns:
        dict{"normal": normal_count, "anomaly": anomaly_count, "total": total_count}

    """
    normal = sum(1 for f in file_list if "normal" in f)
    anomaly = sum(1 for f in file_list if "anomaly" in f)
    return {"normal": normal, "anomaly": anomaly, "total": len(file_list)}


def find_best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """F1 スコアが最大になる閾値を探索して返す.

    BCE sigmoid スコアは「大きいほど異常」なので、
    score > th のときに異常と判定する。

    Args:
        scores (np.ndarray): モデルの出力スコア
        labels (np.ndarray): 真のラベル

    Returns:
        tuple[float, float]: 最適な F1 スコアと閾値
    """
    best_f1, best_th = 0.0, 0.5
    for th in np.arange(0.0, 1.0, 0.01):
        preds = (scores > th).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_th = score, th
    return best_f1, best_th


def main() -> None:
    """メイン関数: 学習と評価の全体の流れを制御."""
    print(f"Using device: {DEVICE}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results", f"supcon_run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"=== All results will be saved to: {save_dir}/ ===\n")

    all_files = [
        f for f in glob(os.path.join(TRAIN_DIR, "*.wav")) if "abundant" not in f
    ]
    if not all_files:
        raise ValueError(
            f"音声ファイルが見つかりません。{TRAIN_DIR} を確認してください。"
        )

    data_counts: dict = {}
    best_thresholds: dict = {}
    model_paths: dict = {}
    all_histories: dict = {}

    # ---- 機種ごとに専門モデルを訓練 ----
    for target_machine in MACHINES:
        print("\n" + "=" * 50)
        print(f" Training Expert Model for: {target_machine}")
        print("=" * 50)

        target_files = [
            f for f in all_files if target_machine in os.path.basename(f)
        ]  # 指定機種のファイルを抽出
        if not target_files:
            print(f"Warning: No data found for {target_machine}. Skipping.")
            continue

        # 8:2 で訓練 / 検証に分割
        val_size = int(len(target_files) * 0.2)
        train_size = len(target_files) - val_size
        train_split, val_split = random_split(
            target_files, [train_size, val_size]
        )  # データセットを分割

        train_files = [target_files[i] for i in train_split.indices]
        val_files = [target_files[i] for i in val_split.indices]

        # データ件数のカウントと表示
        train_counts = count_labels(train_files)
        val_counts = count_labels(val_files)
        data_counts[target_machine] = {"train": train_counts, "val": val_counts}
        print(f"Data - Train: {train_counts} | Val: {val_counts}")

        # クラス不均衡を重みで補正するサンプラー
        num_normal = train_counts["normal"]
        num_anomaly = train_counts["anomaly"]
        weight_normal = 1.0 / num_normal if num_normal > 0 else 0.0
        weight_anomaly = 1.0 / num_anomaly if num_anomaly > 0 else 0.0
        sample_weights = [
            weight_normal if "normal" in f else weight_anomaly for f in train_files
        ]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_loader = create_dataloader(
            file_list=train_files,
            batch_size=BATCH_SIZE,
            is_train=True,  # 学習用のデータローダーはラベルも返す
            sampler=sampler,
        )
        val_loader = create_dataloader(
            file_list=val_files,
            batch_size=BATCH_SIZE,
            shuffle=False,
            is_train=True,
        )

        # モデル、損失関数、最適化アルゴリズムの定義
        model = SupConClassifier(out_dim=EMBEDDING_DIM).to(DEVICE)
        criterion_contrastive = losses.SupConLoss()  # SupConLoss は 同じラベルのサンプルを引き寄せ、異なるラベルのサンプルを遠ざける対照学習の損失関数
        criterion_bce = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        model_save_path = os.path.join(save_dir, f"{target_machine}_best_model.pth")
        best_val_loss = float("inf")
        best_val_auc = 0.0
        epochs_no_improve = 0  # Early Stopping カウンタ

        history_train_loss, history_val_loss, history_f1, history_auc, history_th = (
            [],
            [],
            [],
            [],
            [],
        )

        for epoch in range(EPOCHS):
            # ---- 訓練 ----
            model.train()  # nn.Module を訓練モードに設定（Dropout などが有効になる）
            train_loss = 0.0
            # WeightedRandomSampler を使用しているため、train_loader はシャッフルされている
            for images, binary_labels, _ in train_loader:
                images = images.to(DEVICE)
                labels_cls = binary_labels.to(DEVICE).float().unsqueeze(1)
                labels_con = binary_labels.to(DEVICE)

                embeddings, outputs_cls = model(images)  # 埋め込みと分類スコアを取得
                loss_con = criterion_contrastive(
                    embeddings, labels_con
                )  # 対照学習の損失
                loss_cls = criterion_bce(outputs_cls, labels_cls)  # 分類の損失
                loss = loss_con + loss_cls  # 2つの損失の合計を最終的な損失とする

                optimizer.zero_grad()  # 勾配を初期化
                loss.backward()  # 誤差逆伝播
                optimizer.step()  # パラメータを更新
                train_loss += loss.item()  # バッチの損失を累積

            avg_train_loss = train_loss / len(
                train_loader
            )  # バッチサイズで平均化してエポックごとの損失を算出

            # ---- 検証 ----
            model.eval()  # nn.Module を評価モードに設定（Dropout などが無効になる）
            val_loss = 0.0
            val_scores, val_trues = [], []
            with torch.no_grad():
                for images, binary_labels, _ in val_loader:
                    images = images.to(DEVICE)
                    labels_cls = binary_labels.to(DEVICE).float().unsqueeze(1)
                    labels_con = binary_labels.to(DEVICE)

                    embeddings, outputs_cls = model(images)
                    loss_con = criterion_contrastive(embeddings, labels_con)
                    loss_cls = criterion_bce(outputs_cls, labels_cls)
                    val_loss += (loss_con + loss_cls).item()

                    # BCE の sigmoid スコアを異常確率として記録
                    probs = torch.sigmoid(outputs_cls).cpu().numpy().flatten()
                    val_scores.extend(probs)
                    val_trues.extend(binary_labels.cpu().numpy().flatten())

            avg_val_loss = val_loss / len(val_loader)
            val_scores = np.array(val_scores)
            val_trues = np.array(val_trues)

            # AUC スコアは両クラスが存在する場合にのみ計算する
            epoch_auc = (
                roc_auc_score(val_trues, val_scores)
                if len(np.unique(val_trues)) > 1
                else 0.0
            )
            # BCE スコアを基に F1 スコアが最大になる閾値を探索
            epoch_best_f1, epoch_best_th = find_best_threshold(val_scores, val_trues)

            print(
                f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"AUC: {epoch_auc:.4f} | "
                f"F1: {epoch_best_f1:.4f} (Th: {epoch_best_th:.2f})"
            )

            history_train_loss.append(avg_train_loss)
            history_val_loss.append(avg_val_loss)
            history_f1.append(epoch_best_f1)
            history_auc.append(epoch_auc)
            history_th.append(epoch_best_th)

            # ---- Early Stopping: Val Loss ベースでモデル保存 ----
            # Val Loss が改善したらモデルを保存し、改善しないエポックが続いたら学習を停止する
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_auc = epoch_auc
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(
                        f"Early stopping triggered at epoch {epoch + 1} "
                        f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)"
                    )
                    break

        print(
            f"Finished {target_machine}. "
            f"Best Val Loss: {best_val_loss:.4f} | Best Val AUC: {best_val_auc:.4f}"
        )

        # ベストモデルをロードして検証データで閾値を確定
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        val_scores_final, val_trues_final = [], []
        with torch.no_grad():
            for images, binary_labels, _ in val_loader:
                images = images.to(DEVICE)
                _, outputs_cls = model(images)
                probs = torch.sigmoid(outputs_cls).cpu().numpy().flatten()
                val_scores_final.extend(probs)
                val_trues_final.extend(binary_labels.cpu().numpy().flatten())

        _, best_th = find_best_threshold(
            np.array(val_scores_final), np.array(val_trues_final)
        )
        best_thresholds[target_machine] = best_th
        print(f"  BCE-based threshold: {best_th:.3f}")

        model_paths[target_machine] = model_save_path
        all_histories[target_machine] = {
            "train_loss": history_train_loss,
            "val_loss": history_val_loss,
            "f1": history_f1,
            "auc": history_auc,
        }

        # 学習ログを CSV に保存
        csv_path = os.path.join(save_dir, f"{target_machine}_training_log.csv")
        with open(csv_path, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                ["Epoch", "Train_Loss", "Val_Loss", "Val_F1", "Val_AUC", "Threshold"]
            )
            for i in range(len(history_train_loss)):
                csv_writer.writerow(
                    [
                        i + 1,
                        history_train_loss[i],
                        history_val_loss[i],
                        history_f1[i],
                        history_auc[i],
                        history_th[i],
                    ]
                )

        del model
        torch.cuda.empty_cache()

    # ---- 学習曲線グラフ ----
    print("\nGenerating Combined Learning Curves...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for m in MACHINES:
        if m not in all_histories:
            continue
        h = all_histories[m]
        ep = range(1, len(h["train_loss"]) + 1)
        axes[0, 0].plot(ep, h["train_loss"], marker=".", label=m)
        axes[0, 1].plot(ep, h["val_loss"], marker=".", label=m)
        axes[1, 0].plot(ep, h["f1"], marker=".", label=m)
        axes[1, 1].plot(ep, h["auc"], marker=".", label=m)

    for ax, title in zip(
        axes.flat,
        ["Training Loss", "Validation Loss", "Validation F1", "Validation AUC"],
    ):
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "combined_learning_curves.png"))
    plt.close()

    # ---- 推論と評価 ----
    print("\n=== Routing Evaluation ===")
    all_eval_files = glob(os.path.join(EVAL_DIR, "*.wav"))

    expert_models: dict = {}
    for m in MACHINES:
        if m not in model_paths:
            continue
        model = SupConClassifier(out_dim=EMBEDDING_DIM).to(DEVICE)
        model.load_state_dict(torch.load(model_paths[m]))
        model.eval()
        expert_models[m] = model

    eval_loader = create_dataloader(
        file_list=all_eval_files,
        batch_size=1,
        shuffle=False,
        is_train=False,
    )

    eval_csv_path = os.path.join(save_dir, "evaluation_results_routed.csv")
    eval_summary = {m: {"NORMAL": 0, "ANOMALY": 0, "total": 0} for m in MACHINES}

    with open(eval_csv_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            ["Filename", "MachineID", "AnomalyProb", "AppliedThreshold", "Result"]
        )
        with torch.no_grad():
            for image, m_id, filename in eval_loader:
                m_id = m_id[0] if isinstance(m_id, (list, tuple)) else m_id
                filename = (
                    filename[0] if isinstance(filename, (list, tuple)) else filename
                )

                if m_id not in expert_models:
                    continue

                image = image.to(DEVICE)
                _, outputs = expert_models[m_id](image)
                # BCE の出力に sigmoid を適用して異常確率を得る
                prob = torch.sigmoid(outputs).item()
                th = best_thresholds[m_id]
                pred_label = "ANOMALY" if prob > th else "NORMAL"

                csv_writer.writerow(
                    [filename, m_id, f"{prob:.4f}", f"{th:.3f}", pred_label]
                )
                eval_summary[m_id][pred_label] += 1
                eval_summary[m_id]["total"] += 1

    with open(
        os.path.join(save_dir, "evaluation_summary_routed.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(eval_summary, f, indent=4, ensure_ascii=False)

    print(f"\nAll operations completed! Saved in: {save_dir}/")


if __name__ == "__main__":
    main()
