# kgxnn/explain/gradcam.py
import torch
import torch.nn.functional as F

class GradCAM:
    """
    用法：
      gradcam = GradCAM(backbone, target_layer_name="layer4")
      cam = gradcam(images[i:i+1], forward_fn, target_class=preds[i:i+1])
    其中 forward_fn 是一个函数：forward_fn(images) -> logits  (必须内部走过 backbone 前向，这样钩子能抓到 activations)
    """
    def __init__(self, backbone, target_layer_name="layer4"):
        self.backbone = backbone
        # 取 ResNet 的指定层（默认 layer4）
        modules = dict([*backbone.backbone.named_children()])
        if target_layer_name not in modules:
            raise ValueError(f"No layer '{target_layer_name}' in backbone.backbone. Got: {list(modules.keys())}")
        self.target_layer = modules[target_layer_name]
        self.activations, self.gradients = None, None

        def fwd_hook(m, i, o): self.activations = o
        def bwd_hook(m, gi, go): self.gradients = go[0]
        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, images, forward_fn, target_class=None):
        """
        images: 归一化后的 [B,3,H,W]
        forward_fn: 可调用对象，内部完成 logits = f(images)（必须使用同一个 self.backbone 完成前向）
        target_class: LongTensor/列表/标量；缺省则用 argmax
        """
        # 清梯度
        self.backbone.zero_grad()

        logits = forward_fn(images)  # [B, C]
        if target_class is None:
            target = logits.argmax(dim=1)
        else:
            if isinstance(target_class, int):
                target = torch.tensor([target_class], device=logits.device)
            else:
                target = target_class.to(logits.device)

        one_hot = F.one_hot(target, num_classes=logits.size(1)).float()
        (logits * one_hot).sum().backward()   # 触发 backward，钩子中记录 gradients

        # 计算 CAM
        grads = self.gradients          # [B, C4, H, W]
        acts  = self.activations        # [B, C4, H, W]
        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam.detach()
