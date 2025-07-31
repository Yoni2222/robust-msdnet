# models/msdnet_gated.py
import math
import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Helper modules (unchanged)
# ---------------------------------------------------------------------
class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(nOut),
            nn.ReLU(True)
        )
    def forward(self, x): return self.net(x)

class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, typ: str, bottleneck, bnWidth):
        super().__init__()
        layers, nInner = [], nIn
        if bottleneck:
            nInner = min(nInner, bnWidth * nOut)
            layers += [
                nn.Conv2d(nIn, nInner, 1, bias=False),
                nn.BatchNorm2d(nInner),
                nn.ReLU(True)
            ]
        if typ == "normal":
            layers.append(nn.Conv2d(nInner, nOut, 3, 1, 1, bias=False))
        elif typ == "down":
            layers.append(nn.Conv2d(nInner, nOut, 3, 2, 1, bias=False))
        else:
            raise ValueError
        layers += [nn.BatchNorm2d(nOut), nn.ReLU(True)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class ConvDownNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnW1, bnW2):
        super().__init__()
        self.conv_down   = ConvBN(nIn1, nOut // 2, "down",   bottleneck, bnW1)
        self.conv_normal = ConvBN(nIn2, nOut // 2, "normal", bottleneck, bnW2)
    def forward(self, x):
        res = [x[1], self.conv_down(x[0]), self.conv_normal(x[1])]
        return torch.cat(res, 1)

class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnW):
        super().__init__()
        #self.conv = ConvBN(nIn, nOut, "normal", bottleneck, bnW)
        self.conv_normal = ConvBN(nIn, nOut, "normal", bottleneck, bnW)
    def forward(self, x):
        if not isinstance(x, list): x = [x]
        res = [x[0], self.conv_normal(x[0])]
        return torch.cat(res, 1)

class MSDNFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, args):
        super().__init__()
        self.layers = nn.ModuleList()
        if args.data.startswith("cifar"):
            self.layers.append(ConvBasic(nIn, nOut * args.grFactor[0], 3, 1, 1))
        elif args.data == "ImageNet":
            self.layers.append(nn.Sequential(
                nn.Conv2d(nIn, nOut * args.grFactor[0], 7, 2, 3),
                nn.BatchNorm2d(nOut * args.grFactor[0]),
                nn.ReLU(True),
                nn.MaxPool2d(3, 2, 1)
            ))
        nIn = nOut * args.grFactor[0]
        for i in range(1, args.nScales):
            self.layers.append(ConvBasic(nIn, nOut * args.grFactor[i], 3, 2, 1))
            nIn = nOut * args.grFactor[i]
    def forward(self, x):
        res = []
        for l in self.layers:
            x = l(x); res.append(x)
        return res

class MSDNLayer(nn.Module):
    def __init__(self, nIn, nOut, args, inScales=None, outScales=None):
        super().__init__()
        self.nScales  = args.nScales
        self.inScales = inScales  if inScales  is not None else args.nScales
        self.outScales= outScales if outScales is not None else args.nScales
        self.discard  = self.inScales - self.outScales
        self.offset   = self.nScales - self.outScales
        self.layers   = nn.ModuleList()

        # first output scale
        if self.discard > 0:
            nIn1 = nIn * args.grFactor[self.offset - 1]
            nIn2 = nIn * args.grFactor[self.offset]
            nO   = nOut * args.grFactor[self.offset]
            self.layers.append(ConvDownNormal(nIn1, nIn2, nO, args.bottleneck,
                                               args.bnFactor[self.offset - 1],
                                               args.bnFactor[self.offset]))
        else:
            self.layers.append(
                ConvNormal(nIn * args.grFactor[self.offset],
                           nOut * args.grFactor[self.offset],
                           args.bottleneck, args.bnFactor[self.offset])
            )
        # remaining scales
        for i in range(self.offset + 1, self.nScales):
            nIn1 = nIn * args.grFactor[i - 1]
            nIn2 = nIn * args.grFactor[i]
            nO   = nOut * args.grFactor[i]
            self.layers.append(ConvDownNormal(nIn1, nIn2, nO, args.bottleneck,
                                               args.bnFactor[i - 1],
                                               args.bnFactor[i]))
    def forward(self, x):
        if self.discard > 0:
            inp = [[x[i-1], x[i]] for i in range(1, self.outScales + 1)]
        else:
            inp = [[x[0]]] + [[x[i-1], x[i]] for i in range(1, self.outScales)]
        return [layer(t) for layer, t in zip(self.layers, inp)]

class ParallelModule(nn.Module):
    def __init__(self, mods): super().__init__(); self.m = nn.ModuleList(mods)
    def forward(self, x): return [m(t) for m, t in zip(self.m, x)]

class ClassifierModule(nn.Module):
    def __init__(self, m, c_in, n_cls):
        super().__init__()
        self.m = m
        self.linear = nn.Linear(c_in, n_cls)
    def forward(self, x):
        z = self.m(x[-1])
        z = z.view(z.size(0), -1)
        return self.linear(z)

# ---------------------------------------------------------------------
# Gate head – predicts P(correct) for current exit
# ---------------------------------------------------------------------
class GateHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.g = nn.Sequential(
            nn.Conv2d(in_ch, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )
    def forward(self, feat): return self.g(feat)  # (B,1) logits

# ---------------------------------------------------------------------
# MSDNet + gates
# ---------------------------------------------------------------------
class MSDNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.blocks      = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.gates       = nn.ModuleList()
        self.block_ch    = []
        self.nBlocks     = args.nBlocks

        # --- compute steps per block ---
        self.steps, n_layer_curr, n_total = [args.base], 0, args.base
        for i in range(1, self.nBlocks):
            step = args.step if args.stepmode == "even" else args.step * i + 1
            self.steps.append(step); n_total += step
        print("building network of steps:", self.steps, n_total)

        # --- build blocks ---
        nIn = args.nChannels
        for i in range(self.nBlocks):
            print(f"\n **** Block {i+1} ****")
            block, nIn = self._build_block(nIn, args, self.steps[i],
                                           n_total, n_layer_curr)
            n_layer_curr += self.steps[i]
            self.blocks.append(block)

            cls_in_ch = nIn * args.grFactor[-1]
            self.block_ch.append(cls_in_ch)
            if args.data.startswith("cifar100"):
                self.classifier.append(self._build_classifier_cifar(cls_in_ch, 100))
            elif args.data.startswith("cifar10"):
                self.classifier.append(self._build_classifier_cifar(cls_in_ch, 10))
            elif args.data == "ImageNet":
                self.classifier.append(self._build_classifier_imagenet(cls_in_ch, 1000))
            else:
                raise NotImplementedError

        # --- gate heads ---
        for ch in self.block_ch:
            self.gates.append(GateHead(ch))

        # init weights
        self.apply(self._init_weights)

    # --------------------------- helpers ------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

    def _build_block(self, nIn, args, step, n_total, n_curr):
        layers = [MSDNFirstLayer(3, nIn, args)] if n_curr == 0 else []
        for _ in range(step):
            n_curr += 1
            inSc, outSc = args.nScales, args.nScales
            if args.prune == "min":
                inSc  = min(args.nScales, n_total - n_curr + 2)
                outSc = min(args.nScales, n_total - n_curr + 1)
            elif args.prune == "max":
                interval = math.ceil(n_total / args.nScales)
                inSc  = args.nScales - math.floor(max(0, n_curr - 2) / interval)
                outSc = args.nScales - math.floor((n_curr - 1) / interval)
            else:
                raise ValueError
            layers.append(MSDNLayer(nIn, args.growthRate, args, inSc, outSc))
            print(f"| inScales {inSc} outScales {outSc} "
                  f"inCh {nIn} outCh {args.growthRate} |")
            nIn += args.growthRate
            # transition
            if args.reduction > 0 and (
                (args.prune == "max" and inSc > outSc) or
                (args.prune == "min" and n_curr in {n_total//3, 2*n_total//3})
            ):
                off = args.nScales - outSc
                layers.append(self._build_transition(
                    nIn, int(args.reduction * nIn), outSc, off, args))
                nIn = int(args.reduction * nIn)
        return nn.Sequential(*layers), nIn

    def _build_transition(self, nIn, nOut, outSc, off, args):
        nets = []
        for i in range(outSc):
            nets.append(ConvBasic(nIn*args.grFactor[off+i],
                                  nOut*args.grFactor[off+i], 1, 1, 0))
        return ParallelModule(nets)

    def _build_classifier_cifar(self, nIn, n_cls):
        conv = nn.Sequential(
            ConvBasic(nIn, 128, 3, 2, 1),
            ConvBasic(128, 128, 3, 2, 1),
            nn.AvgPool2d(2)
        )
        return ClassifierModule(conv, 128, n_cls)

    def _build_classifier_imagenet(self, nIn, n_cls):
        conv = nn.Sequential(
            ConvBasic(nIn, nIn, 3, 2, 1),
            ConvBasic(nIn, nIn, 3, 2, 1),
            nn.AvgPool2d(2)
        )
        return ClassifierModule(conv, nIn, n_cls)

    @staticmethod
    def _gather_batch(x_scales, keep_mask):
        return [t[keep_mask] for t in x_scales]

        # -----------------------------------------------------------------
    # Forward with 3 modes
    # -----------------------------------------------------------------
    # def forward(self, x, *, return_gates=False,
    #             use_gates=False, gate_thresh=0.5, return_exit=False):
    #     """
    #     Modes:
    #     • default             → list[Tensor] (all exits)
    #     • return_gates=True   → (cls_list, gate_logit_list)
    #     • use_gates=True      → single logits (early‑exit online)
    #                             + if return_exit=True also returns exit_id
    #     """
    #     assert sum([return_gates, use_gates]) <= 1, \
    #         "Choose only one special mode at a time."
    #
    #     cls_outs, gate_outs = [], []
    #     for i in range(self.nBlocks):
    #         x = self.blocks[i](x)
    #         feat   = x[-1]
    #         g_log  = self.gates[i](feat)          # (B,1)
    #         c_log  = self.classifier[i](x)       # (B,C)
    #         gate_outs.append(g_log); cls_outs.append(c_log)
    #
    #         """if use_gates:
    #             if torch.sigmoid(g_log).ge(gate_thresh).all():
    #                 return (c_log, i) if return_exit else c_log"""
    #         if use_gates:
    #             exit_mask = torch.sigmoid(g_log).ge(gate_thresh).squeeze(1)  # (B,)
    #             if exit_mask.sum() == x.size(0):
    #                 return (c_log, i) if return_exit else c_log
    #
    #     if return_gates:
    #         return cls_outs, gate_outs
    #     if return_exit:                           # fell through to last exit
    #         return cls_outs[-1], self.nBlocks - 1
    #     return cls_outs

    def forward(self, x, *, return_gates=False,
                use_gates=False, gate_thresh=0.5, return_exit=False):
        """
        default           → list[Tensor]             (all exits)
        return_gates=True → (cls_list, gate_list)
        use_gates=True    → per‑sample early‑exit:
                            returns (logits, exit_ids)  if return_exit=True
                            else returns logits (B,C)
        """
        assert sum([return_gates, use_gates]) <= 1

        # ----- vanilla multi‑exit path (unchanged) -----
        if not use_gates:
            cls, gates = [], []
            for i in range(self.nBlocks):
                x = self.blocks[i](x)
                gates.append(self.gates[i](x[-1]))
                cls.append(self.classifier[i](x))
            if return_gates:
                return cls, gates
            return cls

        # ------------- per‑sample early‑exit -------------
        device = x[0].device if isinstance(x, list) else x.device
        B = x[0].size(0) if isinstance(x, list) else x.size(0)

        logits_buffer = [None] * B
        exit_ids = torch.empty(B, dtype=torch.long, device=device)

        active_idx = torch.arange(B, device=device)  # global indices of sub‑batch

        for blk in range(self.nBlocks):
            x = self.blocks[blk](x)
            g_log = self.gates[blk](x[-1])  # (b_active,1)
            c_log = self.classifier[blk](x)  # (b_active,C)

            exit_mask = torch.sigmoid(g_log).squeeze(1) >= gate_thresh  # Bool
            if exit_mask.any():
                done_idx = active_idx[exit_mask]  # global sample ids
                logits_out = c_log[exit_mask]
                for gid, log in zip(done_idx.tolist(), logits_out):
                    logits_buffer[gid] = log
                exit_ids[done_idx] = blk

            # keep undecided samples
            keep_mask = ~exit_mask
            if keep_mask.sum() == 0:  # everyone exited
                active_idx = active_idx.new_empty(0)
                break
            active_idx = active_idx[keep_mask]
            x = self._gather_batch(x, keep_mask)

        # samples that never crossed the threshold take logits from *last* block
        if len(active_idx) > 0:
            final_c_log = self.classifier[-1](x) if blk != self.nBlocks - 1 else c_log
            for gid, log in zip(active_idx.tolist(), final_c_log):
                logits_buffer[gid] = log
            exit_ids[active_idx] = self.nBlocks - 1

        logits_tensor = torch.stack(logits_buffer, 0)  # (B,C)
        if return_exit:
            return logits_tensor, exit_ids
        return logits_tensor

