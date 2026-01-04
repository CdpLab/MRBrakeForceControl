# -*- coding: utf-8 -*-
"""
configs/configs.py

Usage:
    - 修改下面的 DATASET / PATHS 区域即可切换数据与输出目录
    - main.py 中通过：from configs.configs import args 读取配置
"""

from dataclasses import dataclass
from pathlib import Path
import os


def _project_root() -> Path:
    """
    自动定位项目根目录：
    configs/configs.py 的上一级就是 configs，再上一级就是项目根目录
    """
    return Path(__file__).resolve().parent.parent


@dataclass
class Args:
    # =========================
    # 运行模式
    # =========================
    # 1: training, 2: testing
    is_training: int = 1

    # 随机种子与设备
    seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"

    # =========================
    project_root: str = str(_project_root())

    # 数据文件
    data_dir: str = str(_project_root())  # 默认项目根目录
    data_file: str = "force_control_dataset.csv"

    # 结果输出目录
    results_dir: str = str(_project_root() / "fig_GRU_Attention_MRB")

    # 模型保存/加载路径
    checkpoint_dir: str = str(_project_root())
    checkpoint_name: str = "best_model.pth"

    # 日志/可视化输出
    train_fig_dir: str = str(_project_root() / "fig_GRU_Attention_MRB" / "train_results")
    test_fig_dir: str = str(_project_root() / "fig_GRU_Attention_MRB" / "test_results")

    # =========================
    # 数据划分与序列设置
    # =========================
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    seq_len: int = 64
    pred_len: int = 1


    # 举例：输入可能是位移/速度/期望力等，输出可能是电流
    # 如果你代码里是按索引取列，这里可以不填；如果是按列名取，必须对齐 csv。
    input_cols: tuple = ("F",)      # 期望力
    target_col: str = "i_cmd"     # 电流

    # =========================
    # 训练超参数（逆模型：VMD-GRU-Attention）
    # =========================
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0

    # 学习率调度
    use_scheduler: bool = False
    scheduler_step: int = 50
    scheduler_gamma: float = 0.5

    # =========================
    # GRU / LSTM / Attention 模型超参数
    # =========================
    model_name: str = "Attention_GRU"  # "GRU" / "LSTM" / "Attention_GRU"
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1

    # Attention 相关
    attn_dim: int = 128

    # =========================
    # VMD 参数
    # =========================
    use_vmd: bool = True
    vmd_K: int = 5          # 模态数
    vmd_alpha: float = 2000 # 惩罚项
    vmd_tau: float = 0      # 噪声容忍（通常 0）
    vmd_DC: int = 0         # 是否保留直流分量
    vmd_init: int = 1       # 初始化方式
    vmd_tol: float = 1e-7   # 收敛阈值

    # =========================
    # 其它：保存策略
    # =========================
    save_best_only: bool = True
    metric_for_best: str = "rmse"  # "loss" or "rmse"


def get_args() -> Args:
    args = Args()

    env_root = os.getenv("PROJECT_ROOT", "").strip()
    if env_root:
        args.project_root = env_root
        args.data_dir = env_root
        args.results_dir = str(Path(env_root) / "fig_GRU_Attention_MRB")
        args.checkpoint_dir = env_root
        args.train_fig_dir = str(Path(env_root) / "fig_GRU_Attention_MRB" / "train_results")
        args.test_fig_dir = str(Path(env_root) / "fig_GRU_Attention_MRB" / "test_results")

    return args


args = get_args()


if __name__ == "__main__":
    print("is_training:", args.is_training)
    print("project_root:", args.project_root)
    print("data_path:", str(Path(args.data_dir) / args.data_file))
    print("checkpoint_path:", str(Path(args.checkpoint_dir) / args.checkpoint_name))
    print("results_dir:", args.results_dir)
