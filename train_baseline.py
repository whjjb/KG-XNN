# kgxnn/train_baseline.py
import os, argparse, csv, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loaders.cifar import get_loaders
from models.vision_backbones import ResNet50Embed

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        B = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / B))
        return res

def validate(model, criterion, loader, device):
    model.eval()
    loss_sum, top1_sum, top5_sum, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                feats = model(images)            # [B, 2048]
                logits = cls_head(feats)         # [B, 100]
                loss = criterion(logits, labels)
            bsz = labels.size(0)
            acc1, acc5 = accuracy(logits, labels, topk=(1,5))
            loss_sum += loss.item() * bsz
            top1_sum += acc1.item() * bsz / 100.0
            top5_sum += acc5.item() * bsz / 100.0
            n += bsz
    return loss_sum / n, (top1_sum / n) * 100.0, (top5_sum / n) * 100.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    # ★ 新增：日志与 run 管理
    parser.add_argument("--log_dir", type=str, default="./outputs/logs")
    parser.add_argument("--run_name", type=str, default="baseline_resnet50")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    run_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)
    csv_path = os.path.join(run_dir, "training_log.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1) 数据加载
    train_loader, val_loader, num_classes = get_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    # 2) 模型：特征提取器 + 线性头
    model = ResNet50Embed(pretrained=True).to(device)
    global cls_head
    cls_head = nn.Linear(2048, num_classes).to(device)

    # 3) 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        list(model.parameters()) + list(cls_head.parameters()),
        lr=args.lr, weight_decay=0.05
    )
    scaler = GradScaler(device="cuda")

    best_top1 = 0.0

    # 写 CSV 头
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["epoch", "train_loss", "train_top1", "val_loss", "val_top1", "val_top5", "lr"])

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        model.train(); cls_head.train()
        train_loss_sum, train_top1_sum, train_n = 0.0, 0.0, 0

        pbar = tqdm(train_loader, desc=f"Train [{epoch}/{args.epochs}]")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                feats = model(images)
                logits = cls_head(feats)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 累计 train 指标
            bsz = labels.size(0)
            acc1, = accuracy(logits, labels, topk=(1,))
            train_loss_sum += loss.item() * bsz
            train_top1_sum += acc1.item() * bsz / 100.0
            train_n += bsz
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / train_n
        train_top1 = (train_top1_sum / train_n) * 100.0

        val_loss, val_top1, val_top5 = validate(model, criterion, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | top1={val_top1:.2f}% | top5={val_top5:.2f}%")

        # 写 TensorBoard
        cur_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/top1", train_top1, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/top1", val_top1, epoch)
        writer.add_scalar("val/top5", val_top5, epoch)
        writer.add_scalar("optim/lr", cur_lr, epoch)

        # 写 CSV
        with open(csv_path, "a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch, f"{train_loss:.6f}", f"{train_top1:.4f}", f"{val_loss:.6f}", f"{val_top1:.4f}", f"{val_top5:.4f}", f"{cur_lr:.6e}"])

        # 保存最佳
        if val_top1 > best_top1:
            best_top1 = val_top1
            state = {
                "epoch": epoch,
                "backbone": model.state_dict(),
                "cls_head": cls_head.state_dict(),
                "val_top1": val_top1,
                "val_top5": val_top5,
            }
            torch.save(state, os.path.join(args.save_dir, "baseline_resnet50_best.pt"))
            print(f"✅ Saved best checkpoint. Top1={best_top1:.2f}%")

    writer.close()
