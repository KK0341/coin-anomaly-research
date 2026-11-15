"""B5: one shared backbone pass, six class queries over both faces."""
import math
import torch
from torch import nn
from torchvision.models import convnext_tiny


class Dual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = convnext_tiny(weights=None)
        self.net.classifier[2] = nn.Identity()
        self.head = nn.Sequential(nn.Linear(1536, 256), nn.GELU(), nn.Linear(256, 6))
        # Initialize after the B0 modules to preserve their seeded initialization.
        self.queries = nn.Parameter(torch.randn(6, 768) * .02)
        self.evidence_weight = nn.Parameter(torch.randn(6, 768) * .02)
        self.evidence_bias = nn.Parameter(torch.zeros(6))

    def forward(self, x, details=False, uniform=False):
        b = x.shape[0]
        spatial = self.net.features(x.flatten(0, 1))
        global_features = self.net.classifier(self.net.avgpool(spatial))
        global_logits = self.head(global_features.reshape(b, 1536))
        # Reuse the backbone's existing LayerNorm at each spatial position.
        tokens = self.net.classifier[0](spatial).flatten(2).transpose(1, 2)
        tokens = tokens.reshape(b, -1, 768)
        scores = torch.einsum('bnd,kd->bkn', tokens.float(), self.queries.float()) / math.sqrt(768)
        attention = scores.softmax(-1)
        if uniform:
            attention = torch.ones_like(attention) / attention.shape[-1]
        evidence = torch.einsum('bkn,bnd->bkd', attention, tokens.float())
        auxiliary = (evidence * self.evidence_weight.float()).sum(-1) + self.evidence_bias.float()
        logits = global_logits.float() + auxiliary
        if details:
            return logits, global_logits.float(), attention.reshape(b, 6, 2, *spatial.shape[-2:])
        return logits

    def initialize_a8(self, path):
        state = dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
        state.pop('head.2.weight')
        state.pop('head.2.bias')
        result = self.load_state_dict(state, strict=False)
        assert set(result.missing_keys) == {'head.2.weight', 'head.2.bias', 'queries', 'evidence_weight', 'evidence_bias'}
        assert not result.unexpected_keys
