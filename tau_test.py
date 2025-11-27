import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----- prepare synthetic gates (N, E=1) -----
N = 2
# gates_base = torch.rand(N,)
# print("gates_base:", gates_base.flatten().numpy())

def soft_topk(gates, k=1, tau1=5e-2, tau2=1e-1):
    gates_softmax = torch.softmax(gates, dim=-1)
    diff = gates_softmax.unsqueeze(-1) - gates_softmax.unsqueeze(-2)
    sigma = torch.sigmoid(-diff / tau1)
    row_sum = sigma.sum(dim=-1) - 0.5
    r_tilde = 1.0 + row_sum
    eps = 0.5
    a = torch.sigmoid((k + eps - r_tilde) / tau2)
    return a * gates_softmax

# ----- scan tau1, tau2 -----
tau1_list = np.logspace(-3, -1, 200)
tau2_list = np.logspace(-3, -1, 200)
ratio_base = np.linspace(1+1e-5, 10, 200)

ratio1 = np.zeros((len(ratio_base), len(tau1_list)))
for i, ratio_b in enumerate(ratio_base):
    for j, t1 in enumerate(tau1_list):
        gates_base = torch.tensor([0.5, 0.5 / ratio_b])
        g = soft_topk(gates_base, k=1, tau1 = t1, tau2=5e-2)
        g_np = g.detach().numpy().flatten()
        g_sorted = np.sort(g_np)[::-1]
        ratio1[i, j] = g_sorted[0] / (g_sorted[1] + 1e-9)

ratio2 = np.zeros((len(ratio_base), len(tau2_list)))
for i, ratio_b in enumerate(ratio_base):
    for j, t2 in enumerate(tau2_list):
        gates_base = torch.tensor([0.5, 0.5 / ratio_b])
        g = soft_topk(gates_base, k=1, tau1 = 5e-2, tau2=t2)
        g_np = g.detach().numpy().flatten()
        g_sorted = np.sort(g_np)[::-1]
        ratio2[i, j] = g_sorted[0] / (g_sorted[1] + 1e-9)

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=("Ratio 1", "Ratio 2")
)
fig.add_trace(
    go.Surface(z=ratio1.T, x=ratio_base, y=tau1_list),
    row=1, col=1
)
fig.add_trace(
    go.Surface(z=ratio2.T, x=ratio_base, y=tau2_list),
    row=1, col=2
)
fig.update_layout(
    scene = dict(
        xaxis = dict(title='ratio_base'),
        yaxis = dict(title='tau1'),
        zaxis = dict(type='log', title='ratio1')
    ),
    scene2 = dict(
        xaxis = dict(title='ratio_base'),
        yaxis = dict(title='tau2'),
        zaxis = dict(type='log', title='ratio2')
    ),
)

fig.write_html("tau_scan.html")
fig.show()