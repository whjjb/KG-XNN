import os, csv, argparse, json
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets  # 用来拿 CIFAR-100 类名

from data_loaders.cifar import get_loaders
from models.vision_backbones import ResNet50Embed
from models.kg_encoder import KGEncoderGAT
from models.fusion import CrossAttentionFuse
from kg.io import load_kg_csv
from kg.build_graph import make_edge_index

# ⬇️ 用“新版”属性矩阵构建 + 正类加权
from kg.attr_loader import build_attr_matrix, compute_pos_weight

# ⬇️ GloVe 加载工具
from kg.glove_utils import load_glove_as_dict


# --------- 小工具：处理复合词的 GloVe 平均向量 ----------
def get_average_vector_from_parts(words, glove_dict, dim):
    """
    对于 'used_for_transporting_people' 这类复合词，把它拆成 ['used','for','transporting','people']，
    分别查 GloVe，没有就用随机向量，最后取“平均向量”作为该节点的初始化。
    """
    vecs = []
    for w in words:
        vecs.append(glove_dict.get(w, np.random.normal(0, 1, dim)))
    return np.mean(vecs, axis=0)


# --------- Top-k 准确率 ----------
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        B = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
        pred = pred.t()                                                # [maxk, B]
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / B))
        return res


# --------- 验证环节 ----------
@torch.no_grad()
def validate(backbone, cls_head, attr_head, kg_enc, fuse, node_emb, edge_index,
             criterion_cls, loader, device, attr_matrix, pos_weight,
             use_attr_supervision=True, lambda_attr=0.5, lambda_l1=1e-4, use_kg=True):
    """
    验证 loss 组成：
      CE（分类） + lambda_attr * BCE（属性，多标签，可选） + lambda_l1 * L1(attn)（可选）
    其中 BCE 使用 pos_weight（对稀有属性提升权重）。
    """
    backbone.eval(); cls_head.eval(); kg_enc.eval(); fuse.eval(); attr_head.eval()

    total_loss, total_top1, total_top5, total_num = 0.0, 0.0, 0.0, 0

    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device); labels = labels.to(device)
        B = labels.size(0)
        # 祖先广播后的属性目标（固定表）按 batch 索引
        attr_target = attr_matrix[labels] if use_attr_supervision else None

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            # 视觉特征
            v = backbone(imgs)  # [B,2048]

            # KG 融合（可开关）
            if use_kg:
                kg_nodes = kg_enc(node_emb.weight, edge_index)   # [N,256]
                z, attn = fuse(v, kg_nodes)                      # z:[B,256], attn:[B,N]
            else:
                z = torch.zeros(B, 256, device=v.device, dtype=v.dtype)  # 与 fuse 输出维度对齐
                attn = None

            # 拼接后分类
            joint = torch.cat([v, z], dim=-1)            # [B,2048+256]
            class_logits = cls_head(joint)               # [B,num_classes]
            ce = criterion_cls(class_logits, labels)

            total = ce
            if use_attr_supervision:
                # 属性头 + BCE（带 pos_weight）
                attr_logits = attr_head(joint)           # [B,num_attrs]
                bce = nn.functional.binary_cross_entropy_with_logits(
                    attr_logits, attr_target, pos_weight=pos_weight
                )
                total = total + lambda_attr * bce

            if use_kg and (attn is not None) and (lambda_l1 > 0):
                l1 = lambda_l1 * attn.abs().mean(dim=0).sum()
                total = total + l1

        acc1, acc5 = accuracy(class_logits, labels, topk=(1, 5))
        total_loss += total.item() * B
        total_top1 += acc1.item() * B / 100.0
        total_top5 += acc5.item() * B / 100.0
        total_num += B

    avg_loss = total_loss / total_num
    avg_top1 = (total_top1 / total_num) * 100.0
    avg_top5 = (total_top5 / total_num) * 100.0
    return avg_loss, avg_top1, avg_top5


# ============================ 主程序 ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 数据与训练
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)

    # 正则、监督
    parser.add_argument("--lambda_l1", type=float, default=1e-4, help="稀疏正则(解释性)")
    parser.add_argument("--lambda_attr", type=float, default=0.8, help="属性 BCE 权重（建议 0.5~1.0）")
    parser.add_argument("--use_attr_supervision", action="store_true", help="启用属性监督（BCE）")

    # KG 开关（对照实验）
    parser.add_argument("--use_kg", type=bool, default=True)

    # 日志/Run
    parser.add_argument("--log_dir", type=str, default="./outputs/logs")
    parser.add_argument("--run_name", type=str, default="kgxnn_glove_attr_broadcast_pw")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    # KG 节点嵌入
    parser.add_argument("--kg_dim", type=int, default=300, help="KG 节点初始向量维度（与 GloVe 对齐）")
    parser.add_argument("--kg_init", type=str, default="glove", choices=["random", "glove", "glove_frozen"])
    parser.add_argument("--glove_path", type=str, default="./embeddings/glove.6B.300d.txt")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 目录与日志
    os.makedirs(args.save_dir, exist_ok=True)
    run_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=run_dir)
    csv_path = os.path.join(run_dir, "training_log.csv")
    with open(os.path.join(run_dir, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # 数据加载
    train_loader, val_loader, num_classes = get_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    # CIFAR-100 类名（顺序与数据集一致）
    tmp_cifar = datasets.CIFAR100(root=args.data_root, train=False, download=True)
    classes = tmp_cifar.classes

    # ---------------- KG 准备 ----------------
    # 读取 KG 的 nodes/edges
    nodes_df, edges_df, _ = load_kg_csv("./kg")
    # 构建边索引供 GNN 使用
    edge_index = make_edge_index(edges_df, num_nodes=len(nodes_df)).to(device)

    # 🔧 核心修复：从 nodes+edges 构建“祖先广播后的”属性矩阵（而不是旧的 attr.csv）
    attr_matrix, attr_names = build_attr_matrix("./kg", classes, max_up_depth=3, device=device)
    num_attrs = attr_matrix.shape[1]
    print(f"[AttrSupervision] num_attrs={num_attrs}  (from nodes+edges; broadcast depth=3)")

    # 基于类层面出现率计算 pos_weight（长尾属性↑权重）
    pos_weight = compute_pos_weight(attr_matrix, max_weight=20.0).to(device)

    # ---------------- 模型构建 ----------------
    backbone = ResNet50Embed(pretrained=True).to(device)
    kg_enc = KGEncoderGAT(in_dim=args.kg_dim, hid=256, heads=4).to(device)
    fuse = CrossAttentionFuse(v_dim=2048, k_dim=256, out_dim=256).to(device)

    # 节点初始 Embedding（支持 GloVe + 复合词平均）
    node_emb = nn.Embedding(len(nodes_df), args.kg_dim).to(device)
    if args.kg_init in ("glove", "glove_frozen"):
        print(f"[GloVe] Loading from {args.glove_path} (dim={args.kg_dim}) ...")
        w2v = load_glove_as_dict(args.glove_path, dim=args.kg_dim)
        init_mat = np.zeros((len(nodes_df), args.kg_dim))
        covered = 0

        for i, node_name in enumerate(nodes_df["name"].astype(str)):
            if "_" in node_name:
                init_vec = get_average_vector_from_parts(node_name.split("_"), w2v, args.kg_dim)
                # 注意：复合词用平均并不一定都来自词表，此处不计入 coverage
            else:
                init_vec = w2v.get(node_name, np.random.normal(0, 1, args.kg_dim))
                if node_name in w2v:
                    covered += 1
            init_mat[i] = init_vec

        node_emb.weight.data.copy_(torch.from_numpy(init_mat).to(device))
        print(f"[GloVe] coverage(single-token): {covered / len(nodes_df) * 100:.2f}%")

        if args.kg_init == "glove_frozen":
            for p in node_emb.parameters():
                p.requires_grad_(False)
    else:
        nn.init.xavier_uniform_(node_emb.weight)

    # 头部
    cls_head = nn.Linear(2048 + 256, num_classes).to(device)
    attr_head = nn.Linear(2048 + 256, num_attrs).to(device)  # 属性头输出维度与 attr_matrix 列一致

    # 优化器 & 混合精度
    params = (
        list(backbone.parameters()) +
        list(kg_enc.parameters()) +
        list(fuse.parameters()) +
        list(node_emb.parameters()) +
        list(cls_head.parameters()) +
        list(attr_head.parameters())
    )
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    criterion_cls = nn.CrossEntropyLoss()
    scaler = GradScaler(device="cuda")

    best_top1 = 0.0

    # CSV 头
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_top1", "val_loss", "val_top1", "val_top5", "lr"])

    # ---------------- 训练循环 ----------------
    for epoch in range(1, args.epochs + 1):
        backbone.train(); kg_enc.train(); fuse.train(); node_emb.train(); cls_head.train(); attr_head.train()
        use_attr = bool(args.use_attr_supervision) and (args.lambda_attr > 0)

        train_loss_sum, train_top1_sum, train_n = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"Train KG-XNN [{epoch}/{args.epochs}]")

        for imgs, labels in pbar:
            imgs = imgs.to(device); labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                # 视觉特征
                v = backbone(imgs)  # [B,2048]

                # KG 融合（可关）
                if args.use_kg:
                    kg_nodes = kg_enc(node_emb.weight, edge_index)   # [N,256]
                    z, attn = fuse(v, kg_nodes)                      # z:[B,256]
                else:
                    z = torch.zeros(v.size(0), 256, device=v.device, dtype=v.dtype)
                    attn = None

                # 联合特征 & 分类
                joint = torch.cat([v, z], dim=-1)                    # [B,2304]
                class_logits = cls_head(joint)                       # [B,num_classes]
                ce = criterion_cls(class_logits, labels)
                loss = ce

                # 属性监督（BCE + pos_weight）
                if use_attr:
                    attr_target = attr_matrix[labels]                # [B,num_attrs]
                    attr_logits = attr_head(joint)                   # [B,num_attrs]
                    bce = nn.functional.binary_cross_entropy_with_logits(
                        attr_logits, attr_target, pos_weight=pos_weight
                    )
                    loss = loss + args.lambda_attr * bce

                # 稀疏正则（使注意力更“尖锐”，仅在 use_kg 为真时有效）
                if args.use_kg and (attn is not None) and (args.lambda_l1 > 0):
                    l1 = args.lambda_l1 * attn.abs().mean(dim=0).sum()
                    loss = loss + l1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 累计训练指标
            bsz = labels.size(0)
            acc1, = accuracy(class_logits, labels, topk=(1,))
            train_loss_sum += loss.item() * bsz
            train_top1_sum += acc1.item() * bsz / 100.0
            train_n += bsz
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / train_n
        train_top1 = (train_top1_sum / train_n) * 100.0

        # --------- 验证 ---------
        val_loss, val_top1, val_top5 = validate(
            backbone, cls_head, attr_head, kg_enc, fuse, node_emb, edge_index,
            criterion_cls, val_loader, device, attr_matrix, pos_weight,
            use_attr_supervision=use_attr, lambda_attr=args.lambda_attr,
            lambda_l1=args.lambda_l1, use_kg=args.use_kg
        )

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | top1={val_top1:.2f}% | top5={val_top5:.2f}%")

        # TensorBoard
        tb_writer.add_scalar("train/loss", train_loss, epoch)
        tb_writer.add_scalar("train/top1", train_top1, epoch)
        tb_writer.add_scalar("val/loss", val_loss, epoch)
        tb_writer.add_scalar("val/top1", val_top1, epoch)
        tb_writer.add_scalar("val/top5", val_top5, epoch)
        tb_writer.add_scalar("optim/lr", cur_lr, epoch)

        # CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_top1:.4f}",
                             f"{val_loss:.6f}", f"{val_top1:.4f}", f"{val_top5:.4f}", f"{cur_lr:.6e}"])

        # 保存 best
        if val_top1 > best_top1:
            best_top1 = val_top1
            torch.save(
                {
                    "epoch": epoch,
                    "backbone": backbone.state_dict(),
                    "kg_enc": kg_enc.state_dict(),
                    "fuse": fuse.state_dict(),
                    "node_emb": node_emb.state_dict(),
                    "cls_head": cls_head.state_dict(),
                    "attr_head": attr_head.state_dict(),
                    "edge_index": edge_index,
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                    "attr_names": attr_names,
                },
                os.path.join(args.save_dir, "kgxnn_best.pt"),
                # kgxnn_rand_noattr.pt
                # kgxnn_rand_attr.pt
                # kgxnn_glove_noattr.pt
                # kgxnn_best.pt （相当于就是 kgxnn_glove_attr.pt)
            )
            print(f"✅ Saved KG-XNN best. Top1={best_top1:.2f}%")

    tb_writer.close()
