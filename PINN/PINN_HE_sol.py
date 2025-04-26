import numpy as np
import torch
from torch import nn as nn
import matplotlib.pyplot as plt
import analytical_sol as ASOL

device = "cpu"

class FCN(nn.Module):
    "Defines a fully-connected network in PyTorch"
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs =  nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN), 
                        activation()])
        
        self.fch = nn.Sequential(*[ 
                        nn.Sequential(*[ 
                            nn.Linear(N_HIDDEN, N_HIDDEN), 
                            activation()]) for _ in range(N_LAYERS - 1)])
        
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
    
# Model definition (for good intialisation)
def init_weights(m):
    if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
        g = nn.init.calculate_gain('tanh')
        torch.nn.init.xavier_uniform_(m.weight, gain=g)
        # torch.nn.init.xavier_normal_(m.weight, gain=g)
        m.bias.data.fill_(0)

def init_xavier(model, retrain_seed):
    torch.manual_seed(retrain_seed)
    model.apply(init_weights)

def get_loss(x, t):

    ## X_T data distribution
    x_t = torch.cat((x, t), 1) 

    ## Initial condition u(x, 0) = sin(x)
    x_0 = torch.cat((x, 0 * t), 1)

    ## Boundary condition 1: u(0, t) = 0
    bc_1_t = torch.cat((0 * x, t), 1)

    # print(bc_1_t)

    ## Boundary condition 2: u(\pi, t) = 0
    bc_2_t = torch.cat((torch.pi + 0 * x, t), 1)

    u_pinn_out = pinn(x_t)
    
    u_t = torch.autograd.grad(u_pinn_out, t, torch.ones_like(u_pinn_out), create_graph = True)[0]
    u_x = torch.autograd.grad(u_pinn_out, x, torch.ones_like(u_pinn_out), create_graph = True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph = True)[0]

    l_1 = 1.5
    l_2 = 1.5
    l_PDE = 2.0
    l_IC = 1.5
    residual_pde = u_t - u_xx

    residual_bc_1 = pinn(bc_1_t)

    residual_bc_2 = pinn(bc_2_t)

    residual_IC = pinn(x_0) - torch.sin(x)

    loss = ((l_PDE * residual_pde ** 2).mean() + (l_1 * residual_bc_1 ** 2).mean() + (l_2 * residual_bc_2 ** 2).mean() + (l_IC * residual_IC **2).mean())

    return loss


def closure():
    optimizer.zero_grad()
    loss = get_loss(x, t)
    loss.backward()
    return loss

plt.rcParams['figure.figsize'] = [8, 5]
plt.rcParams.update({'font.size': 15})

x = torch.linspace(0, torch.pi, 50, requires_grad=True)  
t = torch.linspace(0, 2, 50, requires_grad=True)

x, t = torch.meshgrid(x, t, indexing="ij")
x = x.reshape(-1, 1).to(device)
t = t.reshape(-1, 1).to(device)

pinn = FCN(2, 1, 42, 5)

retrain = 128
init_xavier(pinn, retrain)
optimizer = torch.optim.LBFGS(pinn.parameters(), lr=0.3)

loss = get_loss(x, t).item()
epochs = 500
threshold = 1.0e-6
i = 0

while (i < epochs -1) and (loss > threshold):
# for i in range(200):
    loss = get_loss(x, t)
    # print(loss)
    optimizer.step(closure)
    # losses.append(loss.cpu().detach().numpy())
    if i % 5 == 0:
        print("Epoch %3d: Current loss: %.10e" % (i, loss.item()))
    i+=1

Nx, Nt = 256, 100
x = torch.linspace(0, torch.pi, Nx)
t = torch.linspace(0, 2.0, Nt)


x, t = torch.meshgrid(x, t, indexing="ij")
x = x.reshape(-1, 1)
t = t.reshape(-1, 1)
x_t = torch.cat((x, t), 1).to(device)

predict = pinn(x_t).reshape(Nx, Nt).cpu()
exact_sol = ASOL.u_analytical()
u_error = predict.detach().numpy() - (exact_sol)

L2 = np.sqrt(np.sum(predict.detach().numpy() - (exact_sol)**2)) / (np.sum(exact_sol**2))

print(f"Relative L2 norm: {L2}")
x = x.reshape(Nx, Nt)
t = t.reshape(Nx, Nt)

u_sol_HEQN = torch.zeros(100, 256)
x_test_HEQN = torch.linspace(0.0, torch.pi, 256).reshape(-1, 1)
t_test_HEQN = torch.zeros(256).reshape(-1, 1)


plt.pcolormesh(x, t, (predict.detach().numpy() - exact_sol)**2, shading='auto', cmap='coolwarm')
plt.colorbar(label=r"$E$")
plt.xlabel(r'$x$')
plt.ylabel(r'$t$')
plt.tight_layout()
plt.title('Error Distribution (Squared Error)')
plt.show()


ax = plt.axes(projection='3d')

ax.plot_surface(x, t, predict.detach().numpy(), rstride=1, cstride=1,
                cmap='coolwarm', edgecolor='none')

ax.set_xlabel(r"$x\;$[m]")
ax.set_ylabel(r"$t\;$[s]")
ax.set_zlabel(r"$T\;$[K]")

ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()



