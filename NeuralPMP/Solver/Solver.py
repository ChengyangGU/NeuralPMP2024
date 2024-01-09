import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def NeuralPMPOffline(horizon, state_dim, action_dim, initial, lr, iters, action_upper, action_lower,
                     TrueDynamics, NeuralDynamics, Terminal, HamiltonianNN, Objective):

    print('--------------------------Solver: NerualPMP(Offline)---------------------------')
    m = horizon
    x0 = initial

    # Control Variable
    # U = [u(0), u(1), ..., u(T-1)]
    u = torch.zeros((m, action_dim), dtype=torch.float32)
    u.requires_grad = True

    # State Variable
    # X = [x(0), x(1), x(2), ..., x(T)]
    xx = torch.zeros((m + 1, state_dim), dtype=torch.float32)
    xx[0] = x0
    xx.requires_grad = True

    # Co-state Variable
    # L = [λ(0), x(1), ..., λ(M-1)]
    # λ(1) = [a(1), b(1)]
    lam = torch.zeros((m, state_dim), dtype=torch.float32)

    # dH/du
    H = torch.zeros((m, action_dim), dtype=torch.float32)

    # Compute time
    t_start = time.time()

    # Loop:
    for it in range(iters):
        # First Compute x
        for i in range(m):
            with torch.no_grad():
                # xx[i+1] = RealDyn(xx[i], u[i])
                tmpipt = torch.concat((xx[i], u[i]), dim=0)
                xx[i + 1] = NeuralDynamics(tmpipt)

        # print('State:', xx)
        # Then Use PMP Conditions: λ（T） = dφ/dxT
        # To reduce graph computations, we create a new tensor xxf
        final = xx[m]
        final = final.detach().numpy()

        xxf = torch.tensor(final, dtype=torch.float32)
        xxf.requires_grad = True

        ff = Terminal(xxf)
        ff.backward()
        lam_T = xxf.grad
        # print(lam_T)

        # From λ(T): Use PMP Conditions to decide λ（t） -- λ（t）= dH/dx + λ(t+1)
        for i in range(m):
            tmp_x = xx[m - 1 - i]
            tmp_u = u[m - 1 - i]

            tmp_x = tmp_x.detach().numpy()
            tmp_u = tmp_u.detach().numpy()

            xxtmp = torch.tensor(tmp_x, dtype=torch.float32)
            xxtmp.requires_grad = True
            uutmp = torch.tensor(tmp_u, dtype=torch.float32)
            uutmp.requires_grad = True
            if i == 0:
                Hx = HamiltonianNN(uutmp, xxtmp, lam_T)
                Hx.backward()
                lam[m - 1 - i] = xxtmp.grad
            else:
                Hx = HamiltonianNN(uutmp, xxtmp, lam[m - i])
                Hx.backward()
                lam[m - 1 - i] = xxtmp.grad

        # Compute dH/du
        for i in range(m):
            u_value = u[i]
            u_value = u_value.detach().numpy()

            utmp = torch.tensor(u_value, dtype=torch.float32)
            utmp.requires_grad = True

            if i == m - 1:
                xtmp = xx[i]
                Ha = HamiltonianNN(utmp, xtmp, lam_T)
                Ha.backward()
                H[i] = utmp.grad
            else:
                xtmp = xx[i]
                Ha = HamiltonianNN(utmp, xtmp, lam[i + 1])
                Ha.backward()
                H[i] = utmp.grad

        # Gradient Descent to Update policy u
        H = torch.where(H > action_upper, action_upper - 0.000001, H)
        H = torch.where(H < action_lower, action_lower - 0.000001, H)

        u = u - lr * H
        if (it + 1) % 100 == 0:
            t_new = time.time()
            print('------------------------Iters:', it + 1, '---------------------------')

            xx_tr = torch.zeros((m + 1, state_dim), dtype=torch.float32)
            xx_tr[0] = x0
            for i in range(m):
                with torch.no_grad():
                    real_next = TrueDynamics(xx_tr[i].detach().numpy(), u[i].detach().numpy())
                    xx_tr[i + 1] = torch.tensor(real_next, dtype=torch.float32)

            print('Optimal Objective:', Objective(xx_tr, u).item())
            print('Compute Time:', t_new - t_start, 's')
            t_start = t_new

    print('Optimal Control:', u.detach().numpy())
    print('Optimal Trajectory:', xx.detach().numpy())

def NeuralPMPOffline_TimeRelatedCost(horizon, state_dim, action_dim, initial, lr, iters, action_upper, action_lower,
                     TrueDynamics, NeuralDynamics, Terminal, HamiltonianNN, Objective):

    print('--------------------------Solver: NerualPMP(Offline, with time-related run cost)---------------------------')
    m = horizon
    x0 = initial

    # Control Variable
    # U = [u(0), u(1), ..., u(T-1)]
    u = torch.zeros((m, action_dim), dtype=torch.float32)
    u.requires_grad = True

    # State Variable
    # X = [x(0), x(1), x(2), ..., x(T)]
    xx = torch.zeros((m + 1, state_dim), dtype=torch.float32)
    xx[0] = x0
    xx.requires_grad = True

    # Co-state Variable
    # L = [λ(0), x(1), ..., λ(M-1)]
    # λ(1) = [a(1), b(1)]
    lam = torch.zeros((m, state_dim), dtype=torch.float32)

    # dH/du
    H = torch.zeros((m, action_dim), dtype=torch.float32)

    # Compute time
    t_start = time.time()

    # Loop:
    for it in range(iters):
        # First Compute x
        for i in range(m):
            with torch.no_grad():
                # xx[i+1] = RealDyn(xx[i], u[i])
                tmpipt = torch.concat((xx[i], u[i]), dim=0)
                xx[i + 1] = NeuralDynamics(tmpipt)

        # print('State:', xx)
        # Then Use PMP Conditions: λ（T） = dφ/dxT
        # To reduce graph computations, we create a new tensor xxf
        final = xx[m]
        final = final.detach().numpy()

        xxf = torch.tensor(final, dtype=torch.float32)
        xxf.requires_grad = True

        ff = Terminal(xxf)
        ff.backward()
        lam_T = xxf.grad
        # print(lam_T)

        # From λ(T): Use PMP Conditions to decide λ（t） -- λ（t）= dH/dx + λ(t+1)
        for i in range(m):
            tmp_x = xx[m - 1 - i]
            tmp_u = u[m - 1 - i]

            tmp_x = tmp_x.detach().numpy()
            tmp_u = tmp_u.detach().numpy()

            xxtmp = torch.tensor(tmp_x, dtype=torch.float32)
            xxtmp.requires_grad = True
            uutmp = torch.tensor(tmp_u, dtype=torch.float32)
            uutmp.requires_grad = True
            if i == 0:
                Hx = HamiltonianNN(uutmp, xxtmp, lam_T, m-1-i)
                Hx.backward()
                lam[m - 1 - i] = xxtmp.grad
            else:
                Hx = HamiltonianNN(uutmp, xxtmp, lam[m - i], m-1-i)
                Hx.backward()
                lam[m - 1 - i] = xxtmp.grad

        # Compute dH/du
        for i in range(m):
            u_value = u[i]
            u_value = u_value.detach().numpy()

            utmp = torch.tensor(u_value, dtype=torch.float32)
            utmp.requires_grad = True

            if i == m - 1:
                xtmp = xx[i]
                Ha = HamiltonianNN(utmp, xtmp, lam_T, i)
                Ha.backward()
                H[i] = utmp.grad
            else:
                xtmp = xx[i]
                Ha = HamiltonianNN(utmp, xtmp, lam[i + 1], i)
                Ha.backward()
                H[i] = utmp.grad

        # Gradient Descent to Update policy u
        H = torch.where(H > action_upper, action_upper - 0.000001, H)
        H = torch.where(H < action_lower, action_lower + 0.000001, H)

        u = u - lr * H
        if (it + 1) % 100 == 0:
            t_new = time.time()
            print('------------------------Iters:', it + 1, '---------------------------')

            xx_tr = torch.zeros((m + 1, state_dim), dtype=torch.float32)
            xx_tr[0] = x0
            for i in range(m):
                with torch.no_grad():
                    real_next = TrueDynamics(xx_tr[i].detach().numpy(), u[i].detach().numpy())
                    xx_tr[i + 1] = torch.tensor(real_next, dtype=torch.float32)

            print('Optimal Objective:', Objective(xx_tr, u).item())
            print('Compute Time:', t_new - t_start, 's')
            t_start = t_new

    print('Optimal Control:', u.detach().numpy())
    print('Optimal Trajectory:', xx.detach().numpy())