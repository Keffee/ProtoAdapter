"""Training objectives and parameter freezing for external OpenOneRec models."""
import math


def component_loss(logits, labels, variants, bridge_mask=None, beta=1.0, gamma=0.0):
    """Shift once, normalize each objective by its own supervised token count."""
    import torch
    import torch.nn.functional as F
    if not all(math.isfinite(x) and x >= 0 for x in (beta, gamma)):
        raise ValueError('Loss weights must be finite and nonnegative')
    if len(variants) != logits.shape[0] or any(v not in {'direct','trace'} for v in variants):
        raise ValueError('Expected a direct/trace variant for every row')
    shifted = labels[:, 1:]
    valid = shifted.ne(-100)
    if not valid.any():
        raise ValueError('No supervised tokens remain after the causal shift')
    losses = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                             shifted.reshape(-1), ignore_index=-100, reduction='none').reshape_as(shifted)
    total = losses.sum() * 0.0
    info = {}
    for variant, weight in [('direct',1.), ('trace',beta)]:
        row_mask = torch.tensor([v == variant for v in variants],device=labels.device)[:,None]
        mask = valid & row_mask
        count = int(mask.sum().item())
        value = losses[mask].mean() if count else losses.sum() * 0.0
        total = total + weight * value
        info[variant + '_tokens'] = count
        info[variant + '_loss'] = float(value.detach())
    mask = torch.zeros_like(valid)
    if bridge_mask is not None:
        trace_rows = torch.tensor([v == 'trace' for v in variants],device=labels.device)[:,None]
        mask = bridge_mask[:,1:].bool() & valid & trace_rows
    count = int(mask.sum().item())
    value = losses[mask].mean() if count else losses.sum() * 0.0
    total = total + gamma * value
    info.update(bridge_tokens=count, bridge_loss=float(value.detach()))
    return total, info


def freeze_backbone(model):
    names = []
    count = 0
    for name, parameter in model.named_parameters():
        enabled = 'embed_tokens' in name or 'lm_head' in name
        parameter.requires_grad_(enabled)
        if enabled:
            names.append(name)
            count += parameter.numel()
    if not count:
        raise ValueError('No OpenOneRec embedding/output parameters found')
    return {'policy': 'embedding_output_only_requires_separate_row_guard', 'trainable_names': names,
            'trainable_parameters': count,
            'total_parameters': sum(p.numel() for p in model.parameters())}


class EmbeddingRowGuard:
    """Single-device equivalent of native frozen-vocabulary row restoration.

    Gradient masking additionally avoids updating frozen rows' optimizer moments.
    Restore after every step, including when an optimizer uses weight decay.
    """
    def __init__(self, model, start_index):
        import torch
        if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 1:
            raise ValueError('start_index must be a positive vocabulary row index')
        self.start_index = start_index
        self.slices = []
        seen = set()
        for module in (model.get_input_embeddings(), model.get_output_embeddings()):
            if module is None or id(module.weight) in seen:
                continue
            weight = module.weight
            if start_index >= weight.shape[0]:
                raise ValueError('start_index must leave trainable vocabulary rows')
            seen.add(id(weight))
            with torch.no_grad():
                self.slices.append((weight, weight[:start_index].detach().clone()))

    def mask_gradients(self):
        for weight, _ in self.slices:
            if weight.grad is not None:
                weight.grad[:self.start_index].zero_()

    def restore(self):
        import torch
        with torch.no_grad():
            for weight, saved in self.slices:
                weight[:self.start_index].copy_(saved)

    def unchanged(self):
        import torch
        return all(torch.equal(weight[:self.start_index], saved) for weight, saved in self.slices)
