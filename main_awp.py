# """
# MSDNet training script **with**
# 1. standard adversarial training (FGSM / PGD‑L∞)
# 2. Adversarial‑Weight Perturbation (AWP) as in Wu et al. 2020
#
# This file is a drop‑in replacement for the original `main.py` that ships with
# https://github.com/kalviny/MSDNet-PyTorch.  All original hyper‑parameters are
# kept; new CLI flags are added at the end of the file (search for *AWP & Adv*
# section).
#
# -----------------------------------------------------------------------
# utils_awp.py   –  the helper from the AWP repository (unchanged)
#
# utils/attack.py –  *optional* helper for PGD/FGSM.  Here we include a **minimal
# inline implementation** to avoid extra files.
# -----------------------------------------------------------------------
#
# Usage example - training
# -------------
# python main_awp.py --data-root ./data --data cifar10 \
#                   --save ./runs/fgsm_awp --arch msdnet \
#                   --epochs 200 --batch-size 128 \
#                   --attack pgd --epsilon 8 --alpha 2 --attack-iters 10 \
#                   --awp-gamma 0.005 --awp-warmup 5
# """


# main_awp.py
from __future__ import print_function, division, absolute_import

import os, sys, math, time, shutil, copy
from argparse import ArgumentParser

#############  ↓↓↓  ORIGINAL IMPORTS  ↓↓↓ #############
from dataloader import get_dataloaders
from adaptive_inference_adv import dynamic_evaluate
import models
from op_counter import measure_model
######################################################

# --- PyTorch ---
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

# --- AWP helper ---
from utils_awp import AdvWeightPerturb
from adv_utils import generate_adv

# Normalised-space helpers
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD  = torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1)
def clamp_normed(x):
    device = x.device
    lo = (0.0 - CIFAR_MEAN.to(device)) / CIFAR_STD.to(device)
    hi = (1.0 - CIFAR_MEAN.to(device)) / CIFAR_STD.to(device)
    return torch.max(torch.min(x, hi), lo)

# ---------------------------------------------------------------------
# Command‑line arguments
# ---------------------------------------------------------------------
def build_parser():
    from args import arg_parser as base_parser
    parser = base_parser

    # adversarial example generation
    adv = parser.add_argument_group('adversarial‑examples')
    adv.add_argument('--attack', default='pgd', choices=['fgsm', 'pgd', 'none'])
    adv.add_argument('--epsilon', type=float, default=8)
    adv.add_argument('--alpha', type=float, default=2)
    adv.add_argument('--attack-iters', type=int, default=10)
    adv.add_argument('--norm', default='l_inf', choices=['l_inf', 'l_2'])

    # AWP
    awp = parser.add_argument_group('AWP')
    awp.add_argument('--awp-gamma', type=float, default=0.005)
    awp.add_argument('--awp-warmup', type=int, default=0)

    # early stopping
    es = parser.add_argument_group('early‑stopping')
    es.add_argument('--early-stop', action='store_true')
    es.add_argument('--patience', type=int, default=20)

    evalg = parser.add_argument_group('evaluation')
    evalg.add_argument('--eval-only', action='store_true')
    evalg.add_argument('--adv-eval', action='store_true')
    evalg.add_argument('--autoattack', action='store_true')
    evalg.add_argument('--dyn-eval', action='store_true')
    evalg.add_argument('--target-p', type=float, default=None)
    evalg.add_argument('--npy-T', type=str, default=None)

    # === NEW === Gate options
    gate = parser.add_argument_group('gates')
    gate.add_argument('--use-gates-infer', action='store_true',
                      help='Use learned gates for online early-exit at eval.')
    gate.add_argument('--gate-thresh', type=float, default=0.5,
                      help='Sigmoid threshold τ for gate decision.')
    gate.add_argument('--gate-lambda', type=float, default=0.2,
                      help='Weight of gate BCE loss.')
    gate.add_argument('--gate-warmup', type=int, default=0,
                      help='Epochs to wait before enabling gate loss.')

    return parser

args = build_parser().parse_args()

# expose CUDA devices early
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# decode string lists
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
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

torch.manual_seed(args.seed)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# scale eps & alpha from /255 to [0,1]
args.epsilon = args.epsilon / 255.0
args.alpha   = args.alpha   / 255.0

# -------------------------------------------------------
# AutoAttack evaluation (unchanged)
# -------------------------------------------------------
def autoattack_eval(data_loader, model, norm, eps):
    from autoattack import AutoAttack
    model.eval()
    xs, ys = [], []
    for x, y in data_loader:
        xs.append(x); ys.append(y)
    x_all = torch.cat(xs, 0).to(DEVICE)
    y_all = torch.cat(ys, 0).to(DEVICE)
    aa_norm = 'Linf' if norm == 'l_inf' else 'L2'
    adversary = AutoAttack(model, norm=aa_norm, eps=eps,
                           version='standard', verbose=True)
    with torch.no_grad():
        adv_outs = adversary.run_standard_evaluation(x_all, y_all, bs=128)
    worst_acc = min(adv_outs.values())
    print('\n=== AutoAttack summary ===')
    for k, acc in adv_outs.items():
        print(f'  Exit {k}: {acc:.2f} %')
    print(f'  Worst-exit     : {worst_acc:.2f} %')

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    best_prec1, best_epoch = 0.0, 0
    epochs_since_best = 0
    os.makedirs(args.save, exist_ok=True)

    sample_model = getattr(models, args.arch)(args)
    n_flops, n_params = measure_model(sample_model, 32 if args.data.startswith('cifar') else 224, 32)
    torch.save(n_flops, os.path.join(args.save, 'flops.pth'))
    del sample_model

    # ---------------- Model ----------------
    model = getattr(models, args.arch)(args)
    model = torch.nn.DataParallel(model).to(DEVICE)

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    # === NEW === gate loss
    gate_criterion = nn.BCEWithLogitsLoss().to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # -------------- AWP set‑up --------------
    proxy       = copy.deepcopy(model)
    proxy_opt   = torch.optim.SGD(proxy.parameters(), lr=0.01)
    awp_adv     = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_opt,
                                   gamma=args.awp_gamma)

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
        ckpt = load_checkpoint(args) if args.evaluate_from is None else \
               torch.load(args.evaluate_from, map_location=DEVICE)
        model.load_state_dict(ckpt['state_dict'])
        print('Loaded checkpoint.  Clean accuracy (last exit):')
        validate(test_loader, model, nn.CrossEntropyLoss().to(DEVICE), gate_criterion)

        if args.adv_eval and args.attack != 'none':
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
        if args.dyn_eval:
            dynamic_evaluate(model, test_loader, val_loader, args)
        sys.exit(0)

    # ---------------- Training loop ----------------
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_prec1, train_prec5, lr = train(
            train_loader, model, criterion, gate_criterion, optimizer, epoch, awp_adv
        )
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, gate_criterion)

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

    print('\n********** Final prediction results **********')
    validate(test_loader, model, criterion, gate_criterion)

# ---------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------
def train(train_loader, model, criterion, gate_criterion, optimizer, epoch, awp_adv):
    batch_time, data_time = AverageMeter(), AverageMeter()
    losses = AverageMeter()
    top1 = [AverageMeter() for _ in range(args.nBlocks)]
    top5 = [AverageMeter() for _ in range(args.nBlocks)]

    model.train()
    end = time.time()
    running_lr = None

    for i, (x, y) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)
        running_lr = lr if running_lr is None else running_lr

        data_time.update(time.time() - end)
        x, y = x.to(DEVICE), y.to(DEVICE)

        # adversarial example
        x_adv = generate_adv(model, x, y,
                             attack=args.attack,
                             norm=args.norm,
                             eps=args.epsilon,
                             alpha=args.alpha,
                             iters=args.attack_iters)

        # AWP perturbation
        diff = None
        if epoch >= args.awp_warmup and args.awp_gamma > 0:
            diff = awp_adv.calc_awp(inputs_adv=x_adv, targets=y)
            awp_adv.perturb(diff)

        # --------------- forward & loss -------------------
        # === CHANGED === request gate logits
        cls_outs, gate_outs = model(x_adv, return_gates=True)

        # CE over exits
        ce_loss = sum(criterion(get_cls(o), y) for o in cls_outs) / len(cls_outs)

        # Gate targets: 1 if exit prediction correct else 0
        with torch.no_grad():
            targets_gate = []
            for o in cls_outs:
                pred_ok = (o.argmax(dim=1) == y).float()
                targets_gate.append(pred_ok)

        if epoch >= args.gate_warmup and args.gate_lambda > 0:
            gate_loss = 0.0
            for g_logit, t in zip(gate_outs, targets_gate):
                gate_loss += gate_criterion(g_logit.squeeze(1), t)
            gate_loss = gate_loss / len(gate_outs)
        else:
            gate_loss = torch.tensor(0.0, device=x_adv.device)

        loss = ce_loss + args.gate_lambda * gate_loss

        # accuracy logging
        for j, out in enumerate(cls_outs):
            prec1, prec5 = accuracy(get_cls(out.data), y, topk=(1, 5))
            top1[j].update(prec1.item(), x.size(0))
            top5[j].update(prec5.item(), x.size(0))

        losses.update(loss.item(), x.size(0))

        # --------------- backward --------------------------
        optimizer.zero_grad()
        loss.backward()
        if diff is not None:
            awp_adv.restore(diff)
        optimizer.step()

        # timing
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i+1}/{len(train_loader)}]\t'
                  f'Time {batch_time.avg:.3f}\t'
                  f'Data {data_time.avg:.3f}\t'
                  f'Loss {losses.val:.4f}\t'
                  f'CE {ce_loss.item():.4f}\t'
                  f'Gate {gate_loss.item():.4f}\t'
                  f'Acc@1 {top1[-1].val:.4f}\t'
                  f'Acc@5 {top5[-1].val:.4f}')

    return losses.avg, top1[-1].avg, top5[-1].avg, running_lr

def validate(val_loader, model, criterion, gate_criterion):
    batch_time, data_time = AverageMeter(), AverageMeter()
    losses = AverageMeter()
    top1 = [AverageMeter() for _ in range(args.nBlocks)]
    top5 = [AverageMeter() for _ in range(args.nBlocks)]
    exit_counter, overall_correct = [0]*args.nBlocks, 0

    n_flops = torch.load(os.path.join(args.save, "flops.pth"))
    model.eval(); end = time.time()

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            data_time.update(time.time() - end)

            # --------------------------------------------------
            # 1) Forward pass
            # --------------------------------------------------
            if args.use_gates_infer:                     # per‑sample EE
                logits, exit_ids = model(
                    x, use_gates=True,
                    gate_thresh=args.gate_thresh,
                    return_exit=True,
                )                     # logits (B,C)  exit_ids (B,)
                loss = criterion(logits, y)              # one set of logits
                losses.update(loss.item(), y.size(0))

                # --------------------------------------------------
                # 2) Per‑sample accuracy accounting
                # --------------------------------------------------
                preds = logits.argmax(1)
                for k in range(args.nBlocks):
                    mask = exit_ids == k
                    if mask.any():
                        correct_k = (preds[mask] == y[mask]).sum().item()
                        top1[k].update(100.0 * correct_k / mask.sum().item(),
                                       mask.sum().item())

                        _, top5_pred = logits[mask].topk(5, 1, True, True)
                        correct5_k = (top5_pred == y[mask].unsqueeze(1))\
                                         .any(1).float().sum().item()
                        top5[k].update(100.0 * correct5_k / mask.sum().item(),
                                       mask.sum().item())

                        exit_counter[k] += mask.sum().item()
                        overall_correct += correct_k

            else:                                        # classic all‑exits
                outs = model(x); outs = outs if isinstance(outs, list) else [outs]
                loss = sum(criterion(o, y) for o in outs) / len(outs)
                losses.update(loss.item(), y.size(0))

                # --------------------------------------------------
                # 2) Exit‑wise accuracy
                # --------------------------------------------------
                for j, o in enumerate(outs):
                    prec1, prec5 = accuracy(o, y, topk=(1,5))
                    top1[j].update(prec1.item(), y.size(0))
                    top5[j].update(prec5.item(), y.size(0))
                    exit_counter[j] += y.size(0)
                    overall_correct += (o.argmax(1) == y).sum().item()

            batch_time.update(time.time() - end); end = time.time()

    # --------------------------------------------------
    # 3) Summary
    # --------------------------------------------------
    total = sum(exit_counter)
    avg_acc = 100.0 * overall_correct / total if total else 0.0

    print("\nSamples per exit:", exit_counter)
    for j in range(args.nBlocks):
        print(f" * Exit {j}: prec@1 {top1[j].avg:.3f}  prec@5 {top5[j].avg:.3f}")
    print(f"Average top‑1 accuracy (all exits combined): {avg_acc:.3f}%")

    # FLOPs report stays the same …
    return losses.avg, top1[-1].avg, top5[-1].avg


def robust_evaluate(data_loader, model, attack_fn):
    model.eval()
    n_exits   = args.nBlocks
    correct   = [0] * n_exits     # # of correct predictions at each exit
    seen      = [0] * n_exits     # # of samples that exited there

    for x, y in data_loader:
        x, y   = x.to(DEVICE), y.to(DEVICE)
        x_adv  = attack_fn(model, x, y)           # FGSM / PGD / …

        with torch.no_grad():

            # ----------------------------------------------------------
            # 1) Per‑sample early‑exit branch
            # ----------------------------------------------------------
            if args.use_gates_infer:
                logits, exit_ids = model(            # ← returns per‑sample exit
                    x_adv,
                    use_gates=True,
                    gate_thresh=args.gate_thresh,
                    return_exit=True
                )                                   # logits  (B,C)
                preds = logits.argmax(1)

                for k in range(n_exits):            # aggregate by exit
                    mask = exit_ids == k
                    if mask.any():
                        correct[k] += (preds[mask] == y[mask]).sum().item()
                        seen[k]    += mask.sum().item()

            # ----------------------------------------------------------
            # 2) Classic “evaluate all exits” branch
            # ----------------------------------------------------------
            else:
                outs = model(x_adv)
                outs = outs if isinstance(outs, list) else [outs]
                for k, o in enumerate(outs):        # same y for every exit
                    pred = o.argmax(1)
                    correct[k] += (pred == y).sum().item()
                    seen[k]    += y.size(0)

    # ----------------------------- summary -----------------------------
    for k in range(n_exits):
        acc = 100.0 * correct[k] / seen[k] if seen[k] else 0.0
        print(f"Robust prec@1 – exit {k}: {acc:5.2f}%  ({correct[k]}/{seen[k]})")

    overall_seen = sum(seen)
    overall_acc  = 100.0 * sum(correct) / overall_seen if overall_seen else 0.0
    print(f"\nAverage robust top‑1 accuracy (all exits): {overall_acc:5.2f}%")

# ---------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self):
        self.val = self.avg = self.sum = 0.0; self.count = 0
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

def adjust_learning_rate(optimizer, epoch, args, batch=None, nBatch=None, method='multistep'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur   = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        lr = args.lr
        if args.data.startswith('cifar'):
            if epoch >= args.epochs * 0.75: lr *= 0.01
            elif epoch >= args.epochs * 0.5: lr *= 0.1
        else:
            lr = args.lr * (0.1 ** (epoch // 30))
    for pg in optimizer.param_groups: pg['lr'] = lr
    return lr

def get_cls(o):
    # works for both (logit, gate_logit) and plain logit
    return o[0] if isinstance(o, (list, tuple)) else o

def save_checkpoint(state, args, is_best, filename):
    ckpt_dir = os.path.join(args.save, 'save_models'); os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, filename)
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, os.path.join(ckpt_dir, 'model_best.pth.tar'))

def load_checkpoint(args):
    latest = os.path.join(args.save, 'save_models', 'model_best.pth.tar')
    return torch.load(latest) if os.path.exists(latest) else None

if __name__ == '__main__':
    main()

