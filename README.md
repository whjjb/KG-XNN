![KG-XNN logo](icon/kgxnn_banner.png)

# KG-XNN: Knowledge-Graph-Guided Explainable Neural Network for Image Classification

KG-XNN is a knowledge-driven, explainable image classification prototype for CIFAR-100. It uses ResNet-50 as the visual backbone,
and fuses visual features with semantic knowledge via a KG encoder (GAT) and cross-attention, with optional attribute supervision and explanations.

KG-XNN 是一个面向 CIFAR-100 的知识驱动可解释图像分类原型。该项目以 ResNet-50 作为视觉主干，
通过知识图谱编码（GAT）与跨注意力融合，将视觉特征与语义知识结合，并支持属性监督与解释输出。


本项目包含 **ResNet50 基线** 与 **KG-XNN（视觉 + 知识图谱）** 两套训练流程，已实现：
- 统一的**训练日志**（CSV + TensorBoard）；
- **可视化脚本**（生成 Loss / Acc 曲线）；
- **GloVe** 词向量用于**知识图谱节点向量初始化**；
- **属性监督**用于将视觉特征与语义属性对齐。
- **五组消融实验**。

> 说明：如果你只做本文的五组消融，`scripts/run_ablation.py` 不是必需的，可以 **删除整个 `scripts/` 目录**；或者保留但只作为个人工具，不纳入论文复现清单。

---

## 目录结构
```
KG-XNN/
├── checkpoints/
│   ├── baseline_resnet50_best.pt
│   ├── kgxnn_glove_noattr.pt
│   ├── kgxnn_rand_attr.pt
│   ├── kgxnn_rand_noattr.pt
│   └── kgxnn_best.pt              # 相当于是kgxnn_glove_attr.pt
├── configs/
├── data/
│   └── cifar-100-python/          # CIFAR-100 数据
├── data_loaders/
│   ├── __init__.py
│   └── cifar.py                   # CIFAR-100 数据加载
├── embeddings/
│   └──glove.6B.300d.txt           # GloVe 预训练向量
├── explain/
│   ├── dump_examples.py
│   ├── gradcam.py
│   ├── kg_path.py         
│   └── plot_training_curves.py    # 可视化曲线脚本
├── kg/
│   ├── attr.csv
│   ├── edges.csv
│   ├── nodes.csv
│   ├── attr_loader.py
│   ├── build_graph.py
│   ├── io.py
│   └── glove_utils.py            # GloVe 加载与节点向量构建
├── models/
│   ├── fusion.py
│   ├── kg_encoder.py
│   └── vision_backbones.py
├── outputs/
│   └── logs/                     # 训练日志（每个 run 一个子目录）
│       ├── baseline_resnet50/
│       ├── kgxnn_rand_noattr/
│       ├── kgxnn_rand_attr/
│       ├── kgxnn_glove_noattr/
│       └── kgxnn_glove_attr/
├── scripts/
│   └──run_ablation.py
├── README.md
├── requirements.txt
├── train_baseline.py             # 基线训练脚本（接入日志/TensorBoard）
└── train_kgxnn.py                # KG-XNN 训练脚本（接入GloVe/日志/消融开关）
```

---

## 环境 & 依赖

- Python ≥ 3.8，PyTorch ≥ 2.0（建议启用 CUDA）
- 依赖列表见 `requirements.txt`，其中包含：
- 数据：CIFAR-100。若本地无数据，`data_loaders/cifar.py` 会负责准备（请按该目录结构放置）。

---

## GloVe 初始化（用于KG节点向量）

- 准备 `glove.6B.300d.txt`（或对应维度的 GloVe 文本向量），放在：
  ```
  embeddings/glove.6B.300d.txt
  ```
- 训练时使用 `--kg_init glove` 并指定 `--glove_path`。未命中的节点（OOV）将随机初始化。

> 提示：在 `--kg_init` 中，脚本支持 `random` 选项（随机初始化）和 `glove` 选项（使用GloVe初始化），也支持 `glove_frozen` 选项（使用Glove初始化，但冻结这些节点向量，不参与训练更新），但**本文消融没有使用`glove_frozen` 选项**。

---

## 训练方式

> 注意：最佳权重保存在checkpoints文件夹下，但是对于不同配置的实验，命名需要在train_kgxnn.py中手动更改。
> 同理，使用dump_examples.py生成示例时，也需在dump_examples.py中手动更改使用的权重文件。

### 1) 基线 ResNet50（无 KG）
```bash
python train_baseline.py \
  --epochs 20 --batch_size 128 \
  --log_dir ./outputs/logs \
  --run_name baseline_resnet50
```
- 最佳权重：`checkpoints/baseline_resnet50_best.pt`
- 日志文件：`outputs/logs/baseline_resnet50/training_log.csv`

### 2) KG-XNN（本研究使用的五组）

> 重要参数：
> - `--kg_init {random, glove}`：节点向量初始化方式（本研究仅用这两种）。
> - `--use_attr_supervision` 与 `--lambda_attr`：是否启用属性监督及其权重。
> - `--lambda_l1`：稀疏/可解释性正则权重（`0` 表示关闭；本研究默认`1e-4`）。
> - `--use_kg`：是否启用KG融合路径（本研究均为`True`）。

- **(A) 纯 ResNet 基线（对照组）**  
  见上文“基线 ResNet50”。

- **(B) 随机初始化 + 无属性监督**
```bash
python train_kgxnn.py \
  --epochs 20 --batch_size 128 \
  --kg_init random \
  --lambda_attr 0 \
  --lambda_l1 1e-4 \
  --log_dir ./outputs/logs \
  --run_name kgxnn_rand_noattr
```

- **(C) 随机初始化 + 属性监督**
```bash
python train_kgxnn.py \
  --epochs 20 --batch_size 128 \
  --kg_init random \
  --use_attr_supervision \
  --lambda_attr 0.8 \
  --lambda_l1 1e-4 \
  --log_dir ./outputs/logs \
  --run_name kgxnn_rand_attr
```

- **(D) GloVe 初始化 + 无属性监督**
```bash
python train_kgxnn.py \
  --epochs 20 --batch_size 128 \
  --kg_init glove \
  --glove_path ./embeddings/glove.6B.300d.txt \
  --lambda_attr 0 \
  --lambda_l1 1e-4 \
  --log_dir ./outputs/logs \
  --run_name kgxnn_glove_noattr
```

- **(E) GloVe 初始化 + 属性监督**
```bash
python train_kgxnn.py \
  --epochs 20 --batch_size 128 \
  --kg_init glove \
  --glove_path ./embeddings/glove.6B.300d.txt \
  --use_attr_supervision \
  --lambda_attr 0.8 \
  --lambda_l1 1e-4 \
  --log_dir ./outputs/logs \
  --run_name kgxnn_glove_attr
```

> 训练完成后，最佳 KG-XNN 权重保存在 `checkpoints/kgxnn_best.pt`。

---

## 日志与可视化

- **TensorBoard**：
  ```bash
  tensorboard --logdir ./outputs/logs
  ```
  在浏览器查看 `train/val` 的 loss、top1、top5、lr 等曲线。

- **CSV 曲线图（PNG）**：
  ```bash
  python explain/plot_training_curves.py \
    --log_csv ./outputs/logs/kgxnn_glove_attr/training_log.csv \
    --out_png ./outputs/logs/kgxnn_glove_attr/curves.png
  ```
  可一次传入多份 `--log_csv` 用于并图对比。

- **CSV 字段**：`epoch, train_loss, train_top1, val_loss, val_top1, val_top5, lr`

---

## 结果展示

**1) Grad-CAM示例**

Grad-CAM 输出的是一张类别相关的显著性热力图（class-discriminative heatmap），用来表达：在当前预测类别下，图像中哪些区域对模型决策贡献最大。热力图中更“热”的区域表示对目标类别得分的正向贡献更大；更“冷”的区域贡献较小或几乎无贡献。

![Grad-CAM Example](outputs/bus_1_overlay.jpg) 
![Grad-CAM Example](outputs/bus_3_overlay.jpg)


**2) 语义解释示例**

```
true=bus
pred=bus
语义层级: bus → vehicle → transport
相关域节点: wheels, road_vehicle, public_transport
注意力命中节点: bus, wheels
```

> 提示：可直接使用 `explain/dump_examples.py` 导出示例，并将图片/文本保存到 `outputs/` 或自定义的 `figures/` 目录。

---

**3) 消融实验**

| 组别 | Run name           |  Top-1(%) |  Top-5(%) |    ECR(%) | KG Init | Attr Sup. | λ_attr | λ_l1 |
| -- | ------------------ | --------: | --------: | --------: | :-----: | :-------: | -----: | ---: |
| A  | baseline_resnet50  | **83.53** | **97.40** |  **0.00** |    –    |     ✗     |      – |    – |
| B  | kgxnn_rand_noattr  | **83.64** | **97.49** |  **0.00** |  random |     ✗     |    0.0 | 1e-4 |
| C  | kgxnn_rand_attr    | **83.68** | **97.39** | **42.50** |  random |     ✓     |    0.5 | 1e-4 |
| D  | kgxnn_glove_noattr | **83.62** | **97.70** | **90.00** |  glove  |     ✗     |    0.0 | 1e-4 |
| E  | kgxnn_glove_attr   | **83.53** | **97.28** | **97.50** |  glove  |     ✓     |    0.5 | 1e-4 |

> ECR（Explanation Coverage Rate）：解释覆盖率。表示测试集中能给出正确语义解释的样本比例——即该样本注意力命中的属性/层级节点中，包含了该类别的真实属性或上位语义。数值越高代表语义解释越充分。

## 常见问题（FAQ）

- **GloVe 覆盖率偏低怎么办？**  
  可尝试更丰富语料的嵌入，或确保节点名用自然短语便于分词命中。

- **维度不匹配？**  
  用 `--kg_dim` 指定与所用 GloVe 维度一致（默认 300），并确保 `glove.6B.300d.txt` 与维度一致。

- **`glove_frozen` 是什么意思？**  
  表示用 GloVe 初始化节点向量后冻结 `node_emb`，训练中不再更新。

- **属性监督没有提升/损失很大？**  
  可调小 `--lambda_attr`（常见范围为 0.3~0.8），或先关闭属性监督排查是否是属性矩阵不匹配。

- **Grad-CAM 可视化很弱或全黑？**  
  先检查是否加载了正确的权重文件（`checkpoints/kgxnn_best.pt`），以及输入是否使用了与训练一致的归一化。

- **训练慢/显存不足？**  
  尝试减小 `--batch_size` 或 `--img_size`，并将 `--num_workers` 调小以降低资源占用。

---

## 引用 / 致谢

- CIFAR-100 dataset: Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
- GloVe: Jeffrey Pennington, Richard Socher, and Christopher D. Manning, *GloVe: Global Vectors for Word Representation*, EMNLP 2014.
- ResNet: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, *Deep Residual Learning for Image Recognition*, CVPR 2016.