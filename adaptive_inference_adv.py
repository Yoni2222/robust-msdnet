# ---------- adaptive_inference_adv.py ----------
import os, math, torch, torch.nn as nn

def dynamic_evaluate(model, test_loader, val_loader, args):
    tester = Tester(model, args)
    cache = os.path.join(args.save,
                         f'logits_{"aa" if args.autoattack else args.attack}.pth')
    if os.path.exists(cache):
        val_pred, val_t, test_pred, test_t = torch.load(cache)
    else:
        val_pred,  val_t  = tester.calc_logits(val_loader)
        test_pred, test_t = tester.calc_logits(test_loader)
        torch.save((val_pred, val_t, test_pred, test_t), cache)

    flops = torch.load(os.path.join(args.save, 'flops.pth'))
    for p in range(1, 40):
        prob = torch.exp(torch.log(torch.tensor(p/20.0)) *
                         torch.arange(1, args.nBlocks+1))
        prob /= prob.sum()
        acc_v, _, T = tester.find_T(val_pred, val_t, prob, flops)
        acc_t, expf = tester.apply_T(test_pred, test_t, flops, T)
        print(f'p={p/20:.2f} | val {acc_v:5.2f}  test {acc_t:5.2f}  '
              f'FLOPs {expf/1e6:6.2f} M')

# ---------- helper ---------------------------------------------------
class Tester:
    def __init__(self, model, args):
        self.m     = model
        self.args  = args
        self.sm    = nn.Softmax(1).cuda()

        from adv_utils import generate_adv             # PGD/FGSM generator
        self.gen_adv = generate_adv

        if args.autoattack:
            from autoattack import AutoAttack
            norm = 'Linf' if args.norm == 'l_inf' else 'L2'
            self.aa = AutoAttack(model, norm=norm,
                                 eps=args.epsilon, version='standard',
                                 verbose=False)

    # --------------- (A) create (possibly) adversarial input ----------
    def _adv(self, x, y):
        if self.args.autoattack:
            with torch.no_grad():
                return self.aa.run_standard_evaluation(x, y, bs=x.size(0))
        if self.args.attack in ('fgsm', 'pgd'):
            return self.gen_adv(self.m, x, y,
                                attack=self.args.attack,
                                norm=self.args.norm,
                                eps=self.args.epsilon,
                                alpha=self.args.alpha,
                                iters=self.args.attack_iters)
        return x      # --attack none

    # --------------- (B) gather per-exit logits -----------------------
    def calc_logits(self, loader):
        self.m.eval()
        n = self.args.nBlocks
        logits = [[] for _ in range(n)]
        targets = []
        for i,(x,y) in enumerate(loader):
            x,y = x.cuda(), y.cuda()
            x   = self._adv(x,y)
            targets.append(y)
            with torch.no_grad():
                out = self.m(x);   out = out if isinstance(out,list) else [out]
                for b in range(n):
                    logits[b].append(self.sm(out[b]))
            if i % self.args.print_freq == 0:
                print(f'logits ({i}/{len(loader)})')

        ts = torch.stack([torch.cat(l,0) for l in logits])      # m×N×C
        return ts, torch.cat(targets,0)

    # --------------- (C) threshold search / application --------------
    def find_T(self, logits, tgt, p, flops):
        m,N,_ = logits.size()
        conf, pred = logits.max(2)
        _, idx = conf.sort(1, True)
        filt = torch.zeros(N, device=logits.device)
        T = torch.full((m,), 1e8, device=logits.device)
        for k in range(m-1):
            keep = math.floor(N*p[k])
            c = 0
            for i in idx[k]:
                if filt[i]==0:
                    c+=1
                    if c==keep:
                        T[k] = conf[k][i]; break
            filt += conf[k].ge(T[k]).float()
        T[-1] = -1e8
        return self._acc_FLOPs(conf,pred,tgt,T,flops)

    def apply_T(self, logits, tgt, flops, T):
        conf,pred = logits.max(2)
        return self._acc_FLOPs(conf,pred,tgt,T,flops)[0:2]

    # helper shared by both ↑
    def _acc_FLOPs(self, conf,pred,tgt,T,flops):
        m,N = conf.size()
        acc_c, take = torch.zeros(m), torch.zeros(m)
        acc = expF = 0
        for i in range(N):
            g = tgt[i]
            for k in range(m):
                if conf[k][i] >= T[k]:
                    if g == pred[k][i]:
                        acc += 1; acc_c[k]+=1
                    take[k]+=1; break
        for k in range(m):
            expF += (take[k]/N)*flops[k]
        return acc*100./N, expF, T
# ---------------------------------------------------------------------
