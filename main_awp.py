"""
MSDNet training script **with**
1. standard adversarial training (FGSM / PGD‑L∞)
2. Adversarial‑Weight Perturbation (AWP) as in Wu et al. 2020

This file is a drop‑in replacement for the original `main.py` that ships with
https://github.com/kalviny/MSDNet-PyTorch.  All original hyper‑parameters are
kept; new CLI flags are added at the end of the file (search for *AWP & Adv*
section).

-----------------------------------------------------------------------
utils_awp.py   –  the helper from the AWP repository (unchanged)

utils/attack.py –  *optional* helper for PGD/FGSM.  Here we include a **minimal
inline implementation** to avoid extra files.
-----------------------------------------------------------------------

Usage example
-------------
python main_awp.py --data-root ./data --data cifar10 \
                  --save ./runs/fgsm_awp --arch msdnet \
                  --epochs 200 --batch-size 128 \
                  --attack pgd --epsilon 8 --alpha 2 --attack-iters 10 \
                  --awp-gamma 0.005 --awp-warmup 5
"""
from __future__ import print_function, division, absolute_import

import os, sys, math, time, shutil, copy
from argparse import ArgumentParser

#############  ↓↓↓  ORIGINAL IMPORTS  ↓↓↓ #############
from dataloader import get_dataloaders
from adaptive_inference import dynamic_evaluate
import models
from op_counter import measure_model
######################################################

# --- PyTorch ---
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

# --- AWP helper ---
from utils_awp import AdvWeightPerturb               # <-- copy from AWP repo

# ---------------------------------------------------------------------
#   Normalised-space helpers  (add just once, close to the other utils)
# ---------------------------------------------------------------------
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD  = torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1)

def clamp_normed(x):
    #Clamp *normalised* tensor to the [0,1] range in RGB space.
    device = x.device
    lo = (0.0 - CIFAR_MEAN.to(device)) / CIFAR_STD.to(device)
    hi = (1.0 - CIFAR_MEAN.to(device)) / CIFAR_STD.to(device)
    return torch.max(torch.min(x, hi), lo)

# ---------------------------------------------------------------------
#   Universal adversarial generator  (works for train + eval branches)
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
#                       Command‑line arguments
# ---------------------------------------------------------------------
# We keep the old ``args.py`` but add a few flags here to stay self‑contained.
# Feel free to migrate these additions back into args.py if you prefer.
# ---------------------------------------------------------------------
def build_parser():
    from args import arg_parser as base_parser
    parser = base_parser

    # adversarial example generation
    adv = parser.add_argument_group('adversarial‑examples')
    adv.add_argument('--attack', default='pgd', choices=['fgsm', 'pgd', 'none'],
                     help='type of input attack (default: pgd)')
    adv.add_argument('--epsilon', type=float, default=8,
                     help='L∞ perturbation budget (in /255 units, default 8)')
    adv.add_argument('--alpha', type=float, default=2,
                     help='PGD step size (in /255 units, default 2)')
    adv.add_argument('--attack-iters', type=int, default=10,
                     help='PGD iterations (default 10)')
    adv.add_argument('--norm', default='l_inf', choices=['l_inf', 'l_2'],
                     help='threat-model norm (default l_inf)')

    # adversarial weight perturbation
    awp = parser.add_argument_group('AWP')
    awp.add_argument('--awp-gamma', type=float, default=0.005,
                     help='multiplier for ||v_l|| ≤ γ ||w_l|| (default 0.005)')
    awp.add_argument('--awp-warmup', type=int, default=0,
                     help='epochs to wait before enabling AWP (default 0)')

    # ---- Early stopping ----
    es = parser.add_argument_group('early‑stopping')
    es.add_argument('--early-stop', action='store_true',
                    help='enable early stopping on val top‑1')
    es.add_argument('--patience', type=int, default=20,
                    help='epochs without improvement before stop (default 20)')

    evalg = parser.add_argument_group('evaluation')
    evalg.add_argument('--eval-only', action='store_true',
                       help='load checkpoint and quit after evaluation')
    evalg.add_argument('--adv-eval', action='store_true',
                       help='when --eval-only, evaluate on adversarial inputs '
                            '(uses --attack / --epsilon / --alpha / --attack-iters)')
    evalg.add_argument('--autoattack', action='store_true',
                       help='run AutoAttack after loading the checkpoint')

    return parser

args = build_parser().parse_args()

# expose CUDA devices early
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# decode string lists from original parser
args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

# dataset class count
if args.data == 'cifar10':
    args.num_classes = 10
elif args.data == 'cifar100':
    args.num_classes = 100
else:
    args.num_classes = 1000

if args.use_valid:
    # train / val / test split like original script
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

# RNG seed
torch.manual_seed(args.seed)

# convenience
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------------
#                    Adversarial example helpers
# ---------------------------------------------------------------------


"""
def fgsm_attack(model, x, y, epsilon):
    #Single‑step FGSM (sign of gradient). epsilon already in [0,1] scale.
    x_adv = x.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        logits = model(x_adv)
        if not isinstance(logits, list):
            logits = [logits]
        # average loss over all exits
        loss = sum(F.cross_entropy(l, y) for l in logits) / len(logits)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv + epsilon * torch.sign(grad)
    return torch.clamp(x_adv, 0, 1).detach()


def pgd_attack(model, x, y, epsilon, alpha, iters):
    #Multi‑step PGD‑L∞. epsilon & alpha already in [0,1].
    x_adv = x + (torch.empty_like(x).uniform_(-epsilon, epsilon))
    x_adv = torch.clamp(x_adv, 0, 1)
    for _ in range(iters):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits = model(x_adv)
            if not isinstance(logits, list):
                logits = [logits]
            loss = sum(F.cross_entropy(l, y) for l in logits) / len(logits)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad)
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
    return x_adv
"""
# scale eps & alpha from /255 to [0,1]
args.epsilon = args.epsilon / 255.0
args.alpha   = args.alpha   / 255.0


# -------------------------------------------------------
#                 AutoAttack evaluation
# -------------------------------------------------------
def autoattack_eval(data_loader, model, norm, eps):
    """
    Runs AutoAttack ('standard' recipe) on the whole loader and
    prints robust accuracy for every exit and the worst exit.

    norm : 'l_inf' | 'l_2'
    eps  : already scaled to [0,1]  (e.g. 8/255)
    """
    from autoattack import AutoAttack
    model.eval()

    # ---- collect the full set once (AutoAttack expects tensors) ----
    xs, ys = [], []
    for x, y in data_loader:
        xs.append(x);  ys.append(y)
    x_all = torch.cat(xs, 0).to(DEVICE)
    y_all = torch.cat(ys, 0).to(DEVICE)

    # ---- configure AA ----
    aa_norm = 'Linf' if norm == 'l_inf' else 'L2'
    adversary = AutoAttack(model, norm=aa_norm, eps=eps,
                           version='standard', verbose=True)

    # AutoAttack will internally split into batches (bs default 128)
    with torch.no_grad():
        adv_outs = adversary.run_standard_evaluation(x_all, y_all, bs=128)

    # adv_outs is a dict {exit_idx : robust_acc}.  Show worst exit:
    worst_acc = min(adv_outs.values())
    print('\n=== AutoAttack summary ===')
    for k, acc in adv_outs.items():
        print(f'  Exit {k}: {acc:.2f} %')
    print(f'  Worst-exit     : {worst_acc:.2f} %')


# ---------------------------------------------------------------------
#                               Main
# ---------------------------------------------------------------------

def main():

    best_prec1, best_epoch = 0.0, 0
    epochs_since_best = 0
    # dirs
    os.makedirs(args.save, exist_ok=True)

    # flop counter (unchanged)
    sample_model = getattr(models, args.arch)(args)
    n_flops, n_params = measure_model(sample_model, 32 if args.data.startswith('cifar') else 224, 32)
    torch.save(n_flops, os.path.join(args.save, 'flops.pth'))
    del sample_model

    # ---------------- Model ----------------
    model = getattr(models, args.arch)(args)
    model = torch.nn.DataParallel(model).to(DEVICE)

    criterion = nn.CrossEntropyLoss().to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # -------------- AWP set‑up --------------
    proxy       = copy.deepcopy(model)
    proxy_opt   = torch.optim.SGD(proxy.parameters(), lr=0.01)
    awp_adv     = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_opt, gamma=args.awp_gamma)

    # resume
    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1      = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    train_loader, val_loader, test_loader = get_dataloaders(args)

    if args.eval_only:
        # load best checkpoint unless --evaluate-from is set
        ckpt = load_checkpoint(args) if args.evaluate_from is None else \
            torch.load(args.evaluate_from, map_location=DEVICE)
        model.load_state_dict(ckpt['state_dict'])
        print('Loaded checkpoint.  Clean accuracy (last exit):')
        validate(test_loader, model, nn.CrossEntropyLoss().to(DEVICE))

        if args.adv_eval and args.attack != 'none':
            """print('\nAdversarial evaluation:')
            if args.attack == 'fgsm':
                fn = lambda m, x, y: fgsm_attack(m, x, y, args.epsilon)
            else:
                fn = lambda m, x, y: pgd_attack(m, x, y,
                                                args.epsilon, args.alpha,
                                                args.attack_iters)
            robust_evaluate(test_loader, model, fn)"""
            print('\nAdversarial evaluation:')
            fn = lambda m, x, y: generate_adv(m, x, y,
                                              attack=args.attack,
                                              norm=args.norm,
                                              eps=args.epsilon,
                                              alpha=args.alpha,
                                              iters=args.attack_iters)
            robust_evaluate(test_loader, model, fn)
        elif args.autoattack:
            autoattack_eval(test_loader, model, norm=args.norm, eps=args.epsilon)
        sys.exit(0)


    # ---------------- Training loop ----------------
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_prec1, train_prec5, lr = train(train_loader, model,
                                                         criterion, optimizer,
                                                         epoch, awp_adv)
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion)

        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1, best_epoch = val_prec1, epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        save_checkpoint({'epoch': epoch,
                         'arch': args.arch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1,
                         'optimizer': optimizer.state_dict()},
                        args, is_best,
                        filename=f'checkpoint_{epoch:03d}.pth.tar')
        print(f'>> Epoch {epoch}: val_prec1={val_prec1:.4f}  (best={best_prec1:.4f} @ {best_epoch})')

        if args.early_stop and epochs_since_best >= args.patience:
          print(f"Early stopping: no improvement for {args.patience} epochs.")
          break
    # final test
    print('\n********** Final prediction results **********')
    validate(test_loader, model, criterion)

# ---------------------------------------------------------------------
#                            Train / Validate
# ---------------------------------------------------------------------

def train(train_loader, model, criterion, optimizer, epoch, awp_adv):
    batch_time, data_time = AverageMeter(), AverageMeter()
    losses = AverageMeter(); top1 = [AverageMeter() for _ in range(args.nBlocks)]
    top5   = [AverageMeter() for _ in range(args.nBlocks)]

    model.train()
    end = time.time()
    running_lr = None

    for i, (x, y) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)
        running_lr = lr if running_lr is None else running_lr

        data_time.update(time.time() - end)
        x, y = x.to(DEVICE), y.to(DEVICE)

        # ---------------- adversarial example ----------------
        x_adv = generate_adv(model, x, y,
                             attack=args.attack,
                             norm=args.norm,
                             eps=args.epsilon,
                             alpha=args.alpha,
                             iters=args.attack_iters)


        # --------------- AWP perturbation ----------------
        diff = None
        if epoch >= args.awp_warmup and args.awp_gamma > 0:
            diff = awp_adv.calc_awp(inputs_adv=x_adv, targets=y)
            awp_adv.perturb(diff)

        # --------------- forward & loss -------------------
        outputs = model(x_adv)
        if not isinstance(outputs, list):
            outputs = [outputs]
        loss = sum(criterion(o, y) for o in outputs) / len(outputs)  # average exits

        # accuracy logging
        for j, out in enumerate(outputs):
            prec1, prec5 = accuracy(out.data, y, topk=(1, 5))
            top1[j].update(prec1.item(), x.size(0))
            top5[j].update(prec5.item(), x.size(0))

        losses.update(loss.item(), x.size(0))

        # --------------- backward --------------------------
        optimizer.zero_grad()
        loss.backward()
        if diff is not None:
            awp_adv.restore(diff)    # bring weights back to centre
        optimizer.step()

        # --------------- timing ----------------------------
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i+1}/{len(train_loader)}]\t'
                  f'Time {batch_time.avg:.3f}\t'
                  f'Data {data_time.avg:.3f}\t'
                  f'Loss {losses.val:.4f}\t'
                  f'Acc@1 {top1[-1].val:.4f}\t'
                  f'Acc@5 {top5[-1].val:.4f}')

    return losses.avg, top1[-1].avg, top5[-1].avg, running_lr

# validate() is unchanged from original main.py (copied verbatim below)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter(); losses = AverageMeter(); data_time = AverageMeter()
    top1 = [AverageMeter() for _ in range(args.nBlocks)]
    top5 = [AverageMeter() for _ in range(args.nBlocks)]

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            """if args.data.startswith('cifar'):
                mean = torch.tensor([0.4914, 0.4822, 0.4465], device=DEVICE).view(1, 3, 1, 1)
                std = torch.tensor([0.2471, 0.2435, 0.2616], device=DEVICE).view(1, 3, 1, 1)
                x = x * std + mean"""
            data_time.update(time.time() - end)

            outputs = model(x)
            if not isinstance(outputs, list):
                outputs = [outputs]
            loss = sum(criterion(o, y) for o in outputs) / len(outputs)
            losses.update(loss.item(), x.size(0))

            for j, out in enumerate(outputs):
                prec1, prec5 = accuracy(out.data, y, topk=(1, 5))
                top1[j].update(prec1.item(), x.size(0))
                top5[j].update(prec5.item(), x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print(f'Epoch: [{i+1}/{len(val_loader)}]\t'
                      f'Time {batch_time.avg:.3f}\t'
                      f'Data {data_time.avg:.3f}\t'
                      f'Loss {losses.val:.3f}\t'
                      f'Acc@1 {top1[-1].val:.3f}\t'
                      f'Acc@5 {top5[-1].val:.3f}')

    for j in range(args.nBlocks):
        print(f' * Exit {j}: prec@1 {top1[j].avg:.3f}  prec@5 {top5[j].avg:.3f}')

    worst_exit = min(t.avg for t in top1)  # worst (lowest) prec@1
    print(f'Worst-exit prec@1: {worst_exit:.3f}')
    return losses.avg, top1[-1].avg, top5[-1].avg

def robust_evaluate(data_loader, model, attack_fn):
    """Accuracy on adversarial examples only (no clean pass)."""
    """model.eval()
    correct, total = 0, 0
    for x, y in data_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv = attack_fn(model, x, y)         # FGSM or PGD
        with torch.no_grad():
            outs = model(x_adv)
            if not isinstance(outs, list):
                outs = [outs]
            logits = outs[-1]                  # last exit
            pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    print(f'Robust top-1 accuracy: {correct/total*100:.2f} %  '
          f'({correct}/{total})')"""

    model.eval()

    n_exits   = args.nBlocks          # number of MSDNet exits
    correct   = [0] * n_exits         # per-exit correct counts
    total     = 0

    for x, y in data_loader:
        x, y   = x.to(DEVICE), y.to(DEVICE)
        x_adv  = attack_fn(model, x, y)          # FGSM / PGD etc.

        with torch.no_grad():
            outs = model(x_adv)
            if not isinstance(outs, list):       # safety for 1-exit model
                outs = [outs]

            for j, logits in enumerate(outs):
                pred = logits.argmax(dim=1)
                correct[j] += (pred == y).sum().item()

        total += y.size(0)

    # ---- reporting ----
    for j, c in enumerate(correct):
        print(f'Robust prec@1 – exit {j}: {100.0*c/total:5.2f}%  ({c}/{total})')

    worst_correct  = min(correct)
    last_correct   = correct[-1]
    worst_prec1    = 100.0 * worst_correct / total
    last_prec1     = 100.0 * last_correct  / total

    print(f'\nRobust top-1 accuracy (worst exit): {worst_prec1:5.2f}%  '
          f'({worst_correct}/{total})')
    print(f'Robust top-1 accuracy (last  exit):  {last_prec1:5.2f}%  '
          f'({last_correct}/{total})')


# ---------------------------------------------------------------------
#                     Misc (AverageMeter, accuracy, etc.)
# ---------------------------------------------------------------------
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# learning‑rate schedule unchanged from original

def adjust_learning_rate(optimizer, epoch, args, batch=None, nBatch=None, method='multistep'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur   = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        lr = args.lr
        if args.data.startswith('cifar'):
            if epoch >= args.epochs * 0.75:
                lr *= 0.01
            elif epoch >= args.epochs * 0.5:
                lr *= 0.1
        else:
            lr = args.lr * (0.1 ** (epoch // 30))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr

# checkpoint helpers (identical to original but path logic simplified)

def save_checkpoint(state, args, is_best, filename):
    ckpt_dir = os.path.join(args.save, 'save_models'); os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, filename)
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, os.path.join(ckpt_dir, 'model_best.pth.tar'))

def load_checkpoint(args):
    latest = os.path.join(args.save, 'save_models', 'model_best.pth.tar')
    return torch.load(latest) if os.path.exists(latest) else None

# ---------------------------------------------------------------------
if __name__ == '__main__':
    main()
