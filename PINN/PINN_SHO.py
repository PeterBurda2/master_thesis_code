import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

# device = "cpu"


def exact_sol(d, w0, t):
    assert d < w0
    w = np.sqrt(w0 ** 2 - d **2)
    phi = np.arctan(-d / w)
    A = 1.0 / (2.0 * np.cos(phi))
    cos = torch.cos(phi + (w * t))
    exp = torch.exp(-d * t)
    u = exp * 2.0 * A * cos
    return u

class FCN(nn.Module):
    "Defines a fully-connected network in PyTorch"
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super(FCN, self).__init__()
        activation = nn.Tanh
        self.fcs =  nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN), 
                        activation()]).to(device)
        
        self.fch = nn.Sequential(*[ 
                        nn.Sequential(*[ 
                            nn.Linear(N_HIDDEN, N_HIDDEN), 
                            activation()]) for _ in range(N_LAYERS - 1)]).to(device)
        
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT).to(device)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

def get_loss(t_bc, t_domain):

    lambda1 = 1e-2
    lambda2 = 1e-3

    # lambda1 = 1.0
    # lambda2 = 1.0

    ##BC LOSS FUN
    u_BC = pinn(t_bc)
    loss1 = torch.mean(torch.squeeze((u_BC - 1.0) ** 2))
    du_BC = torch.autograd.grad(u_BC, t_bc, torch.ones_like(u_BC), create_graph = True)[0]
    loss2 = torch.mean(torch.squeeze((du_BC - 0.0) ** 2))

    u_ODE = pinn(t_domain)
    du_ODE = torch.autograd.grad(u_ODE, t_domain, torch.ones_like(u_ODE), create_graph = True)[0]
    du2_ODE = torch.autograd.grad(du_ODE, t_domain, torch.ones_like(u_ODE), create_graph = True)[0]
    loss3 = torch.mean((du2_ODE + (mu * du_ODE)+ (k*u_ODE)) ** 2)



    # print(lambda1 * loss2, lambda2 * loss3)
    loss = loss1 + (lambda1 * loss2) + (lambda2 * loss3)

    return loss

def closure():
    loss = get_loss(t_boundary, t_physics)
    optimiser.zero_grad()
    loss.backward()
    return loss

def init_weights(m):
    if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
        g = nn.init.calculate_gain('tanh')
        torch.nn.init.xavier_uniform_(m.weight, gain=g)
        #torch.nn.init.xavier_normal_(m.weight, gain=g)
        m.bias.data.fill_(0)

def init_xavier(model, retrain_seed):
    torch.manual_seed(retrain_seed)
    model.apply(init_weights)

retrain  = 128
plt.rcParams['figure.figsize'] = [8, 5]
plt.rcParams.update({'font.size': 15})

# torch.cuda.memory._record_memory_history()
pinn = FCN(1, 1, 32, 4)
optimiser = torch.optim.LBFGS(pinn.parameters(),lr = 0.25)


init_xavier(pinn, retrain)


t_boundary = torch.tensor(0.0, requires_grad=True).view(-1, 1).to(device)
t_physics = torch.linspace(0.0, 1.0, 30, requires_grad=True).view(-1, 1).to(device)

d = 2.0
w0 = 20.0
mu = 2.0 * d
k = w0 ** 2
Epoch_number = 500

t_test = torch.linspace(0.0, 1.0, 300).view(-1, 1).to(device)
L2_log = np.zeros(Epoch_number, dtype=np.float32)

u_exact = exact_sol(d, w0, t_test)
LOSS = get_loss(t_boundary, t_physics)
threshold = 1.0e-5
i = 0

print(LOSS.item())

##LEARNING SSH
st = time.time()

while (i < Epoch_number -1)  and (LOSS.item() > threshold): 
# for i in range(Epoch_number):
    LOSS = get_loss(t_boundary, t_physics)
    L2_norm = np.sum((pinn(t_test)[:, 0].cpu().detach().numpy() - exact_sol(d, w0, t_test)[:, 0].cpu().detach().numpy()) ** 2) / np.sum(exact_sol(d, w0, t_test)[:, 0].cpu().detach().numpy()**2)
    L2_log[i] = L2_norm
    
    optimiser.step(closure)
    print(f"Epoch: {i}")
    print(f"{LOSS.item():.16f}")

    if i % 15 == 0:
        u = pinn(t_test)
        plt.plot(t_physics[:, 0].cpu().detach().numpy(), torch.zeros_like(t_physics)[:, 0].cpu().detach().numpy(), 'bo', alpha=0.6, label = 'Training time points')
        plt.plot(t_boundary[:,0].cpu().detach().numpy(), torch.zeros_like(t_boundary)[:, 0].cpu().detach().numpy(), 'ro', alpha=0.8, label = 'Traning boundary point')
        plt.plot(t_test[:, 0].cpu().detach().numpy(), u_exact[:, 0].cpu().detach().numpy(), marker = 'x', linestyle = '-',color = 'grey', label = 'Exact solution')
        plt.plot(t_test[:, 0].cpu().detach().numpy(), u[:, 0].cpu().detach().numpy(),  linestyle = "-", color = "g", label = f"PINN Solution")
        plt.title(f"Training epoch {i}")
        plt.xlabel(f"$t$ [s]")
        plt.ylabel(f"$x$ [m]")
        plt.legend()
        plt.tight_layout()
        plt.show()
    i+=1

end = time.time()

print(f"Elapsed time GPU: {end - st}")

u_sol_HO = torch.zeros(100, 1).reshape(-1, 1)
t_sol_HO = torch.zeros(100, 1).reshape(-1, 1)

# t_sol = torch.zeros(1, 1).reshape(-1, 1)
t_test_HO = torch.linspace(0.0, 1.5, 300).view(-1, 1).to(device)


# %%
fig, axs = plt.subplots()
axs.plot(t_test_HO[:, 0].cpu().detach().numpy(), exact_sol(d, w0, t_test_HO)[:, 0].cpu().detach().numpy(), marker = 'x', linestyle = '-',color = 'grey', label = 'Exact solution')
axs.plot(t_test_HO[:, 0].cpu().detach().numpy(), pinn(t_test_HO)[:, 0].cpu().detach().numpy(),  linestyle = "-", color = "g", label = f"PINN Solution")
axs.set_xlabel(f"$t$ [s]")
axs.set_ylabel(f"$x$ [m]")
axs.fill_between(t_test[:, 0].cpu().detach().numpy(), -0.75, 1,
                color='blue', alpha=0.2, label = f"Inside of the training domain")
axs.fill_between(torch.linspace(1.0, 1.5, 300).view(-1, 1)[:, 0].cpu().detach().numpy(), -0.75, 1,
                color='red', alpha=0.2, label = f"Outside of the training domain")

plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
plt.tight_layout()
plt.show()
