from __future__ import division, print_function
import torch, torch.nn.functional as F

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
CIFAR_STD  = torch.tensor([0.2471, 0.2435, 0.2616]).view(1,3,1,1)

def clamp_normed(x):
    #Clamp *normalised* tensor to the [0,1] range in RGB space.
    device = x.device
    lo = (0.0 - CIFAR_MEAN.to(device)) / CIFAR_STD.to(device)
    hi = (1.0 - CIFAR_MEAN.to(device)) / CIFAR_STD.to(device)
    return torch.max(torch.min(x, hi), lo)

def generate_adv(model, x, y, *, attack, norm, eps, alpha, iters):
    """
    Return adversarial version of x under the chosen threat model.
    All eps/alpha are already on the [0,1] scale.
    """
    if attack == 'none':
        return x

    if norm == 'l_inf':
        def linf_step(img, grad, step):
            return clamp_normed(img + step * grad.sign())

        x_adv = x.clone().detach()
        if attack == 'fgsm':
            #x_adv.requires_grad_()
            x_adv = x.clone().detach().requires_grad_(True)
            #loss = F.cross_entropy(model(x_adv)[-1], y)
            outs = model(x_adv)
            if not isinstance(outs, list):        # single-exit safety
                outs = [outs]
            loss = sum(F.cross_entropy(o, y) for o in outs) / len(outs)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = clamp_normed(x_adv + eps * grad.sign()).detach()
            return linf_step(x_adv, grad, eps)

        # ----- PGD-L∞ -----
        x_adv += torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = clamp_normed(x_adv)
        for _ in range(iters):
            x_adv.requires_grad_(True)
            #loss = F.cross_entropy(model(x_adv)[-1], y)
            outs = model(x_adv)
            if not isinstance(outs, list):        # single-exit safety
                outs = [outs]
            loss = sum(F.cross_entropy(o, y) for o in outs) / len(outs)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = linf_step(x_adv.detach(), grad, alpha)
            x_adv = clamp_normed(torch.max(torch.min(x_adv, x + eps), x - eps))
        return x_adv

    # ---------------- PGD-L2 ----------------
    B = x.size(0)
    delta = torch.randn_like(x).view(B, -1)
    delta = delta / (delta.norm(2, dim=1, keepdim=True) + 1e-8)
    x_adv = clamp_normed(x + delta.view_as(x) * eps)

    for _ in range(iters):
        x_adv.requires_grad_(True)
        #loss = F.cross_entropy(model(x_adv)[-1], y)
        outs = model(x_adv)
        if not isinstance(outs, list):  # single-exit safety
            outs = [outs]
        loss = sum(F.cross_entropy(o, y) for o in outs) / len(outs)
        grad = torch.autograd.grad(loss, x_adv)[0]
        g_flat = grad.view(B, -1)
        g_norm = g_flat.norm(2, dim=1, keepdim=True) + 1e-8
        x_adv = clamp_normed(x_adv.detach() + alpha * (g_flat / g_norm).view_as(x_adv))

        delta = (x_adv - x).view(B, -1)
        d_norm = delta.norm(2, dim=1, keepdim=True).clamp(min=1e-8)
        factor = torch.min(torch.ones_like(d_norm), eps / d_norm)
        #x_adv = clamp_normed((x + delta * factor).view_as(x_adv))
        delta = (delta * factor).view_as(x)  # (B,3,32,32)
        x_adv = clamp_normed(x + delta)
    return x_adv
