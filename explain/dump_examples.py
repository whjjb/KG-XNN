import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.amp import autocast
from torchvision import transforms, datasets

from data_loaders.cifar import get_loaders
from models.vision_backbones import ResNet50Embed
from models.kg_encoder import KGEncoderGAT
from models.fusion import CrossAttentionFuse
from kg.io import load_kg_csv
from kg.build_graph import make_edge_index
from explain.gradcam import GradCAM
from explain.kg_path import KGPathExtractor


def overlay_heatmap(img_t, cam_t):
    """
    把 Grad-CAM 热力图叠加到原图上，输出一张彩色可读图。
    img_t: [3,H,W] 模型输入用的图像(标准化后)
    cam_t: Grad-CAM 输出 (可以是[1,1,h,w] 或 [1,h,w] 或 [h,w])

    返回: uint8 的 (H,W,3) RGB 图像，可直接 cv2.imwrite
    """
    # 1. 反归一化到人眼可读
    denorm = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )
    img = denorm(img_t.clone()).clamp(0, 1)  # [3,H,W]
    img_np = img.permute(1, 2, 0).cpu().numpy()  # [H,W,3] in [0,1]
    H, W = img_np.shape[:2]

    # 2. 处理CAM形状，拉成2D热度图
    cam_arr = cam_t.detach().cpu().squeeze()
    cam_arr = cam_arr.numpy().astype(np.float32)
    cam_arr = cv2.resize(cam_arr, (W, H))

    # 3. 归一化到0-255并上伪彩色
    cam_arr = cam_arr - cam_arr.min()
    cam_arr = cam_arr / (cam_arr.max() + 1e-6)
    cam_uint8 = (cam_arr * 255).astype(np.uint8)

    heat = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)  # BGR
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # -> RGB, [0,1]

    # 4. 融合(0.8原图 + 0.2热力)，让主体尽量清晰
    overlay = 0.8 * img_np + 0.2 * heat
    overlay = np.clip(overlay, 0, 1)
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    return overlay_uint8


if __name__ == "__main__":
    os.makedirs("./outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device (in dump_examples): {device}")

    # ============ 1. 加载数据 (验证集) ============
    # 我们用较大 batch_size 来尽快遇到目标类别
    _, val_loader, num_classes = get_loaders(
        data_root="./data",
        batch_size=128,
        num_workers=2,
        img_size=224,
    )

    # CIFAR-100 的类名（下标 -> 类名）
    tmp = datasets.CIFAR100(root="./data", train=False, download=True)
    classes = tmp.classes  # e.g. ["apple", "aquarium_fish", ..., "train", ...]
    name_to_idx = {name: idx for idx, name in enumerate(classes)}

    # 我们想重点展示/解释的类别（论文里会放图的那批）
    target_names = [
        #"tiger", "wolf", "seal", "whale", "fox", "dolphin"    # 动物/哺乳/水生哺乳
        #"fox", "dolphin"                                      # 动物/哺乳/水生哺乳（使用）
        #"bus", "truck", "train", "bicycle", "rocket", "tank"  # 交通工具
        "bus", "train"                                       # 交通工具(使用）
        #"rose", "tulip", "pine_tree",                         # 植物/花/树
    ]
    # 过滤出CIFAR中真实存在的类
    target_names = [n for n in target_names if n in name_to_idx]

    # 每个类最多保存几张图（防止dump太多）
    max_per_class = 20
    saved_per_class = {n: 0 for n in target_names}

    # ============ 2. 载入训练好的最优权重 ============
    ckpt = torch.load("./checkpoints/kgxnn_best.pt", map_location=device)
    # kgxnn_rand_noattr.pt
    # kgxnn_rand_attr.pt
    # kgxnn_glove_noattr.pt
    # kgxnn_best.pt （相当于就是 kgxnn_glove_attr.pt)

    # 视觉主干
    backbone = ResNet50Embed(pretrained=True).to(device)
    backbone.load_state_dict(ckpt["backbone"])

    # 知识图谱编码模块
    kg_enc = KGEncoderGAT(in_dim=300, hid=256, heads=4).to(device)
    kg_enc.load_state_dict(ckpt["kg_enc"])

    # Cross-Attention 融合模块
    fuse = CrossAttentionFuse(v_dim=2048, k_dim=256, out_dim=256).to(device)
    fuse.load_state_dict(ckpt["fuse"])

    # KG 节点的可训练向量
    # 注意：这里我们恢复的是 node_emb，而不是重新init
    node_emb = nn.Embedding(
        ckpt["node_emb"]["weight"].shape[0],
        ckpt["node_emb"]["weight"].shape[1],
    ).to(device)
    node_emb.load_state_dict(ckpt["node_emb"])

    # 分类头：从 joint_feat -> CIFAR100 logits
    cls_head = nn.Linear(2048 + 256, num_classes).to(device)
    cls_head.load_state_dict(ckpt["cls_head"])

    # 属性头（虽然这脚本里不直接用它画图，但我们load进来保证一致性）
    if "attr_head" in ckpt:
        attr_head = nn.Linear(
            2048 + 256,
            ckpt["attr_head"]["weight"].shape[0]
        ).to(device)
        attr_head.load_state_dict(ckpt["attr_head"])
    else:
        attr_head = None  # 向后兼容旧checkpoint

    # 知识图谱边
    edge_index = ckpt["edge_index"].to(device)

    # 载入KG的结构（节点/边）
    nodes_df, edges_df, _ = load_kg_csv("./kg")

    # 用于人类可读解释（语义层级 + 关键属性）
    kge = KGPathExtractor(nodes_df, edges_df)

    # Grad-CAM 钩子，挂在视觉 backbone 的最后一层卷积（默认 layer4）
    gradcam = GradCAM(backbone)

    # ============ 3. 定义 forward_fn 给 Grad-CAM 调用 ============
    # Grad-CAM 需要一个“我给你一张图 -> 你算出分类logits”的函数。
    # 我们希望这个函数和训练路径完全一致 (v -> kg -> z -> concat -> cls_head)
    def forward_fn(x_batch):
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            v_ = backbone(x_batch)                             # [B,2048]
            kg_nodes_ = kg_enc(node_emb.weight, edge_index)    # [N,256]
            z_, _attn_ = fuse(v_, kg_nodes_)                   # z_:[B,256]
            joint_ = torch.cat([v_, z_], dim=-1)               # [B,2304]
            logits_ = cls_head(joint_)                         # [B,num_classes]
        return logits_

    # ============ 4. 遍历验证集，挑目标类别的样本并导出可视化 ============
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向：拿到分类预测 + 注意力 (attn) 以便提取关键知识节点
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            v = backbone(images)                               # [B,2048]
            kg_nodes_vec = kg_enc(node_emb.weight, edge_index) # [N,256]
            z, attn = fuse(v, kg_nodes_vec)                    # z:[B,256], attn:[B,N]
            joint_feat = torch.cat([v, z], dim=-1)             # [B,2304]
            logits = cls_head(joint_feat)                      # [B,num_classes]
            preds = logits.argmax(dim=1)                       # [B]

        B = images.size(0)
        for i in range(B):
            true_idx = int(labels[i].item())
            true_name = classes[true_idx]

            # 只导出我们关心的类
            if true_name not in saved_per_class:
                continue
            if saved_per_class[true_name] >= max_per_class:
                continue

            # 注释掉以下三行以跳过注意力热力图生成
            img_i = images[i:i+1]  # [1,3,H,W]
            cam_map = gradcam(img_i, forward_fn, target_class=preds[i:i+1])
            overlay_img = overlay_heatmap(images[i].cpu(), cam_map[0])

            # 改为直接使用原始图像（反归一化后）
            # denorm = transforms.Normalize(
            #     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            #     std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            # )
            # img = denorm(images[i].clone()).clamp(0, 1)  # [3,H,W]
            # img_np = img.permute(1, 2, 0).cpu().numpy()  # [H,W,3] in [0,1]
            # overlay_img = (img_np * 255).astype(np.uint8)

            # 解释知识注意力：
            # 1. 取该样本的 cross-attention 分布 attn[i]  ( [N_nodes] )
            # 2. 找注意力最高的若干KG节点，交给解释器
            attn_i = attn[i].detach().cpu()  # [N_nodes]
            topk_idx = torch.topk(attn_i, k=3).indices.numpy().tolist()

            pred_name = classes[int(preds[i].item())]

            # kge.describe 会返回类似：
            # "语义层级: pine_tree → conifer → tree → plant；关键属性: has_petals, is_conifer"
            explain_text = kge.describe(pred_name, topk_idx)

            # 输出文件名示例： pine_tree_0_overlay.jpg / pine_tree_0_explain.txt
            out_idx = saved_per_class[true_name]
            img_out_path = f"./outputs/{true_name}_{out_idx}_overlay.jpg"
            txt_out_path = f"./outputs/{true_name}_{out_idx}_explain.txt"

            # 保存叠加热图
            cv2.imwrite(
                img_out_path,
                cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)  # OpenCV写文件用BGR
            )

            # 保存文本解释
            with open(txt_out_path, "w", encoding="utf-8") as f:
                f.write(f"true={true_name}\n")
                f.write(f"pred={pred_name}\n")
                f.write(explain_text + "\n")

            print(f"[Saved] {true_name} #{out_idx} -> {img_out_path}")
            saved_per_class[true_name] += 1

        # 如果所有目标类都够了，就提前退出
        if all(saved_per_class[n] >= max_per_class for n in saved_per_class):
            break

    # 在主循环之后增加逐样本统计逻辑
    print("\n📊 逐样本解释统计:")

    # 初始化统计变量
    total_samples = 0
    explained_samples = 0
    class_sample_stats = {name: {'total': 0, 'explained': 0} for name in target_names}

    # 遍历每个目标类别和每个样本
    for name in target_names:
        for idx in range(max_per_class):
            txt_path = f"./outputs/{name}_{idx}_explain.txt"
            if os.path.exists(txt_path):
                # 增加该类别的样本计数
                class_sample_stats[name]['total'] += 1
                # 增加总样本计数
                total_samples += 1

                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 检查是否有注意力命中
                    if "注意力命中节点: 无" not in content:
                        class_sample_stats[name]['explained'] += 1
                        explained_samples += 1

    # 输出每个类别的详细统计
    for name, stats in class_sample_stats.items():
        if stats['total'] > 0:
            class_coverage = stats['explained'] / stats['total']
            print(f"- {name}: {stats['explained']}/{stats['total']} 样本被解释 ({class_coverage:.2%})")

    # 输出总体统计
    overall_coverage = explained_samples / total_samples if total_samples > 0 else 0
    print(f"\n📈 总体统计:")
    print(f"- 总共导出样本数: {total_samples}")
    print(f"- 成功解释的样本数: {explained_samples}")
    print(f"- 总体解释覆盖率: {overall_coverage:.2%}")

    print("✅ 已导出目标类别的可视化结果到 ./outputs/ 目录下。")

    # Explanation Coverage Rate:
    # 衡量模型为预测样本提供有效知识图谱解释的能力，计算公式为：能生成有效KG路径解释的样本数 / 总评估样本数