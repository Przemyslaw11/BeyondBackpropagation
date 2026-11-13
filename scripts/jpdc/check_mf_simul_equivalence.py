# Check: one Adam over lambda-weighted detached losses == per-layer Adams (Algorithm 1, K=1)
# with hidden-layer weight_decay/lambda and eps/lambda. Mirrors bp_ds._detached_intermediate_activations.
import torch, torch.nn.functional as F
torch.manual_seed(0); torch.set_default_dtype(torch.float64)
D, H, C, L, lam, wd, lr = 20, 16, 5, 3, 0.0207, 2.66e-6, 2.3e-4
def init():
    g = torch.Generator().manual_seed(1)
    Ws = [torch.randn(D if i == 0 else H, H, generator=g) * .3 for i in range(L)]
    Ms = [torch.randn(D if i == 0 else H, C, generator=g) * .3 for i in range(L + 1)]  # M_0 reads x
    return [t.requires_grad_() for t in Ws], [t.requires_grad_() for t in Ms]
def losses(Ws, Ms, x, y):
    a, out = x, [F.cross_entropy(x @ Ms[0], y)]
    for i in range(L):
        a = torch.relu(a.detach() @ Ws[i]); out.append(F.cross_entropy(a @ Ms[i + 1], y))
    return out  # out[L] is the readout (weight 1), out[0..L-1] auxiliary (weight lam)
data = [(torch.randn(32, D), torch.randint(0, C, (32,))) for _ in range(50)]
# A: MF-Simul as implemented
WA, MA = init(); optA = torch.optim.Adam(WA + MA, lr=lr, weight_decay=wd)
# B: Algorithm 1 with per-layer Adam; layer k = (W_k, M_k), M_0 alone
WB, MB = init()
groups = [[MB[0]]] + [[WB[i], MB[i + 1]] for i in range(L)]
optB = [torch.optim.Adam(g, lr=lr, weight_decay=wd if k == L else wd / lam, eps=1e-8 if k == L else 1e-8 / lam)
        for k, g in enumerate(groups)]
for x, y in data:
    la = losses(WA, MA, x, y); tot = la[L] + lam * sum(la[:L])
    optA.zero_grad(); tot.backward(); optA.step()
    lb = losses(WB, MB, x, y)
    for k, o in enumerate(optB): o.zero_grad(); lb[k].backward(); o.step()
diff = max((a - b).abs().max().item() for a, b in zip(WA + MA, WB + MB))
print("max param diff after 50 steps:", diff); assert diff < 1e-10
