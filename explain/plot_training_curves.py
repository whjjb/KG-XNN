# explain/plot_training_curves.py
"""
用法：
python explain/plot_training_curves.py --log_csv ./outputs/logs/kgxnn_glove_attr/training_log.csv \
                                      --out_png ./outputs/logs/kgxnn_glove_attr/curves.png
或批量传多个 --log_csv 进行对比（图例用 run 目录名的最后一层）。
"""
import os, argparse, csv
import matplotlib.pyplot as plt

def load_csv_metrics(path):
    xs, train_loss, train_top1, val_loss, val_top1, val_top5 = [], [], [], [], [], []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            xs.append(int(r["epoch"]))
            train_loss.append(float(r["train_loss"]))
            train_top1.append(float(r["train_top1"]))
            val_loss.append(float(r["val_loss"]))
            val_top1.append(float(r["val_top1"]))
            val_top5.append(float(r["val_top5"]))
    return xs, train_loss, train_top1, val_loss, val_top1, val_top5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_csv", type=str, nargs="+", required=True, help="一个或多个 CSV 日志")
    parser.add_argument("--out_png", type=str, default=None, help="输出曲线图路径")
    args = parser.parse_args()

    plt.figure(figsize=(9, 7))
    # 子图1：Loss
    for csv_path in args.log_csv:
        xs, tr_l, _, va_l, _, _ = load_csv_metrics(csv_path)
        label = os.path.basename(os.path.dirname(csv_path))
        plt.plot(xs, tr_l, linestyle="-", label=f"{label} train_loss")
        plt.plot(xs, va_l, linestyle="--", label=f"{label} val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Training & Validation Loss")
    plt.legend(); plt.grid(True, alpha=0.3)
    if args.out_png:
        os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
        plt.savefig(args.out_png, dpi=150)
    plt.show()

    plt.figure(figsize=(9, 7))
    # 子图2：Accuracy
    for csv_path in args.log_csv:
        xs, _, tr_a, _, va_a, va5 = load_csv_metrics(csv_path)
        label = os.path.basename(os.path.dirname(csv_path))
        plt.plot(xs, tr_a, linestyle="-", label=f"{label} train_top1")
        plt.plot(xs, va_a, linestyle="--", label=f"{label} val_top1")
        plt.plot(xs, va5, linestyle=":", label=f"{label} val_top5")
    plt.xlabel("epoch"); plt.ylabel("accuracy (%)"); plt.title("Top-1/Top-5 Accuracy")
    plt.legend(); plt.grid(True, alpha=0.3)
    if args.out_png:
        base, ext = os.path.splitext(args.out_png)
        out2 = base + "_acc.png"
        plt.savefig(out2, dpi=150)
    plt.show()
