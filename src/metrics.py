import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = torch.argmax(logits, dim=-1)
    return (preds == targets).float().mean()


def mae(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(preds - targets))


def mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean((preds - targets) ** 2)


def rmse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(mse(preds, targets))


def classification_prf(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preds = torch.argmax(logits, dim=-1)
    num_classes = int(torch.max(targets).item()) + 1 if targets.numel() > 0 else 0
    if num_classes <= 0:
        zero = torch.tensor(0.0, device=logits.device)
        return zero, zero, zero
    eps = torch.tensor(1e-12, device=logits.device)
    precision_list = []
    recall_list = []
    f1_list = []
    for cls in range(num_classes):
        pred_pos = preds == cls
        true_pos = targets == cls
        tp = (pred_pos & true_pos).sum().float()
        fp = (pred_pos & ~true_pos).sum().float()
        fn = (~pred_pos & true_pos).sum().float()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    precision = torch.stack(precision_list).mean()
    recall = torch.stack(recall_list).mean()
    f1 = torch.stack(f1_list).mean()
    return precision, recall, f1
