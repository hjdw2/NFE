import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, types

def snip_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias, self.stride, self.padding, self.dilation, self.groups)

def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)

def SNIP(net, sparsity, train_dataloader):
    keep_ratio = 1 - sparsity
    inputs, targets = next(iter(train_dataloader))
    inputs = inputs.cuda()
    targets = targets.cuda()

    net = copy.deepcopy(net)

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

    net.zero_grad()
    outputs = net.forward(inputs)
    loss = F.nll_loss(outputs, targets)
    loss.backward()

    grads_abs = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            grads_abs.append(torch.abs(layer.weight_mask.grad))

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])

    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())

    return (keep_masks)
