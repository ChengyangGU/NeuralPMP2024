import numpy as np
import time
import math
import torch
import torch.optim as optim
import torch.nn.functional as func
from NeuralDynamics import Networks
from Solver import Solver

# Environment 1: LQR
class LQR():
    def __init__(self):
        # Parameters Settings for True LQR System

        # Dimensions
        self.state_dim = 5
        self.action_dim = 3

        # Time Horizon
        self.T = 10

        # Initial State
        self.x0 = np.array([0, 0, 1, 1, 0])

        # Dynamic Matrices
        self.A = np.array([[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1]])
        self.B = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [1, 1, 0],
              [0, 1, 1]])

        # State cost matrix
        self.Q = np.eye(self.state_dim)

        # Control cost matrix
        self.R = np.eye(self.action_dim)

        # Terminal cost matrix
        self.QT = np.array([[5, 0, 0, 0, 0],
              [0, 4, 0, 0, 0],
              [0, 0, 2, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 3]])

        # Action Bounds
        self.action_lower = -100000
        self.action_upper = 100000

        # Dataset (features, labels)
        self.DF = []
        self.DL = []

        # Neural Networks for Dynamics
        self.NeuralDyn = Networks.NNTypeB(input=self.state_dim+self.action_dim, output=self.state_dim, hidden=64)

    # True Dynamics of the System
    def TrueDynamics(self, x, u):
        next = np.dot(self.A, x) + np.dot(self.B, u)
        return next

    # Run Cost Function:
    def RunCost(self, x, u):
        x = x.unsqueeze(1)
        u = u.unsqueeze(1)

        Q = torch.tensor(self.Q, dtype=torch.float32)
        R = torch.tensor(self.R, dtype=torch.float32)

        tmp_x = torch.mm(Q, x)
        cost_x = torch.dot(tmp_x.squeeze(1), x.squeeze(1))

        tmp_u = torch.mm(R, u)
        cost_u = torch.dot(tmp_u.squeeze(1), u.squeeze(1))

        output = cost_x + cost_u
        return output

    # Terminal Cost Function
    def Terminal(self, x):
        x = x.unsqueeze(1)
        QT = torch.tensor(self.QT, dtype=torch.float32)

        tmp_x = torch.mm(QT, x)
        cost_x = torch.dot(tmp_x.squeeze(1), x.squeeze(1))

        output = cost_x
        return output

    # Overall Objective
    def Obj(self, xx, uu):
        path_cost = 0
        for t in range(self.T):
            path_cost = path_cost + self.RunCost(xx[t], uu[t])

        final_cost = self.Terminal(xx[self.T])

        total_cost = path_cost + final_cost

        return total_cost

    # Training Neural Dynamics
    def TrainNeuralDynamics(self, sample_num, train_iters):
        # Sampling
        for i in range(sample_num):
            x_cur = -5 * np.ones(self.state_dim) + 10 * np.random.rand(self.state_dim)
            u_cur = -5 * np.ones(self.action_dim) + 10 * np.random.rand(self.action_dim)

            input = np.concatenate((x_cur, u_cur), axis=0)
            output = self.TrueDynamics(x_cur, u_cur)

            self.DF.append(input)
            self.DL.append(output)

        X_train = np.array(self.DF, dtype=np.float32)
        y_train = np.array(self.DL, dtype=np.float32)

        X_train_T = torch.from_numpy(X_train)
        y_train_T = torch.from_numpy(y_train)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.NeuralDyn.to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

        print('NN Dynamics Pre-train.......')
        train_start = time.time()

        for ep in range(train_iters):
            model.train()
            data, target = X_train_T.to(device), y_train_T.to(device)
            optimizer.zero_grad()
            output = model(data)

            # output = output.view(output.size(0))
            loss = func.mse_loss(output, target)
            # loss = GradientLoss(data, output, target, alpha=1, beta=1)
            loss.backward()
            optimizer.step()
            if (ep + 1) % 1000 == 0:
                print('Epoch:', ep + 1, 'Loss = ', loss.item())
        print('Pre-training Time---------------------', time.time() - train_start, 's')
        self.NeuralDyn = model

    def HamiltonianNeural(self, u, x, lam):
        input = torch.concat((x, u), dim=0)
        output = self.RunCost(x, u) + torch.dot(lam, self.NeuralDyn(input))
        return output

    # Solve for optimal control (Default: NeuralPMPOffline--lr=0.001, iters=3000)
    def Solve(self, sample_num, train_iters, solver='NeuralPMPOffline', PMPlr=1e-3, PMP_iters=3000):

        self.TrainNeuralDynamics(sample_num=sample_num, train_iters=train_iters)

        if solver == 'NeuralPMPOffline':
            x0 = torch.tensor(self.x0, dtype=torch.float32)
            Solver.NeuralPMPOffline(horizon=self.T, state_dim=self.state_dim, action_dim=self.action_dim, initial=x0,
                                    lr=PMPlr, iters=PMP_iters, action_lower=self.action_lower, action_upper= self.action_upper,
                                    TrueDynamics=self.TrueDynamics, NeuralDynamics=self.NeuralDyn, Terminal=self.Terminal,
                                    HamiltonianNN=self.HamiltonianNeural, Objective=self.Obj
                                    )
        elif solver == 'NeuralPMPOffline_TimeRelated':
            x0 = torch.tensor(self.x0, dtype=torch.float32)
            Solver.NeuralPMPOffline_TimeRelatedCost(horizon=self.T, state_dim=self.state_dim, action_dim=self.action_dim, initial=x0,
                                    lr=PMPlr, iters=PMP_iters, action_lower=self.action_lower, action_upper= self.action_upper,
                                    TrueDynamics=self.TrueDynamics, NeuralDynamics=self.NeuralDyn,
                                    Terminal=self.Terminal,
                                    HamiltonianNN=self.HamiltonianNeural, Objective=self.Obj
                                    )


# Environment 2: Battery
class Battery():
    def __init__(self):
        # Parameters Settings for True Battery System

        # Dimensions
        self.state_dim = 1
        self.action_dim = 1

        # Time Horizon
        self.T = 24

        # Initial State
        self.x0 = np.array([[2]], dtype=np.float32)

        # Target Final State
        self.xf = np.array([[3]], dtype=np.float32)

        # Time Varying Price Signal
        self.p = np.zeros((self.T))
        for i in range(0, self.T):
            if 0 <= i <= 7:
                self.p[i] = 5
            elif 8 <= i <= 12:
                self.p[i] = 10
            elif 13 <= i <= 17:
                self.p[i] = 7
            else:
                self.p[i] = 6

        # Penalty for action
        self.alpha = 0.1

        # Penalty for state outside bounds
        self.beta = 200

        # Penalty for terminal state
        self.gamma = 200

        # State Bounds
        self.State_lower = 0
        self.State_upper = 10

        # Action Bounds
        self.action_lower = -5
        self.action_upper = 5


        # Dataset (features, labels)
        self.DF = []
        self.DL = []

        # Neural Networks for Dynamics
        self.NeuralEff = Networks.NNTypeB(input=self.action_dim, output=self.state_dim, hidden=64)


    # True Efficiency ζ(u) of the system
    def Sigmoid(self, u):
        return 1 / (1 + np.exp(-u))

    def TrueEfficiency(self, u):
        eff = self.Sigmoid(-u) + 0.5
        return eff

    # True Dynamics of the system
    def TrueDynamics(self, x, u):
        next = self.TrueEfficiency(u) * u + x
        return next

    # Run Cost Function:
    def RunCost(self, x, u, t):

        ctrl_cost = self.p[t] * u
        ctrl_penal = self.alpha * u * u

        if x[0] < self.State_lower:
            state_penal = self.beta * (x - self.State_lower) * (x - self.State_lower)
        elif x[0] > self.State_upper:
            state_penal = self.beta * (x - self.State_upper) * (x - self.State_upper)
        else:
            state_penal = 0

        output = ctrl_cost + ctrl_penal + state_penal
        return output

    # Terminal Cost Function
    def Terminal(self, x):
        xf = torch.tensor(self.xf, dtype=torch.float32)
        output = self.gamma * (x - xf[0]) * (x - xf[0])
        return output

    # Overall Objective
    def Obj(self, xx, uu):
        path_cost = 0
        for t in range(self.T):
            path_cost = path_cost + self.RunCost(xx[t], uu[t], t)

        final_cost = self.Terminal(xx[self.T])

        total_cost = path_cost + final_cost

        return total_cost

    def TrainNeuralEfficiency(self, sample_num, train_iters):
        # Sampling
        for i in range(sample_num):
            u_cur = -5 * np.ones(self.action_dim) + 10 * np.random.rand(self.action_dim)

            input = u_cur
            output = self.TrueEfficiency(input)

            self.DF.append(input)
            self.DL.append(output)

        X_train = np.array(self.DF, dtype=np.float32)
        y_train = np.array(self.DL, dtype=np.float32)

        X_train_T = torch.from_numpy(X_train)
        y_train_T = torch.from_numpy(y_train)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.NeuralEff.to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

        print('NN Dynamics Pre-train.......')
        train_start = time.time()

        for ep in range(train_iters):
            model.train()
            data, target = X_train_T.to(device), y_train_T.to(device)
            optimizer.zero_grad()
            output = model(data)

            # output = output.view(output.size(0))
            loss = func.mse_loss(output, target)
            # loss = GradientLoss(data, output, target, alpha=1, beta=1)
            loss.backward()
            optimizer.step()
            if (ep + 1) % 1000 == 0:
                print('Epoch:', ep + 1, 'Loss = ', loss.item())
        print('Pre-training Time---------------------', time.time() - train_start, 's')
        self.NeuralEff = model

    # Neural Dynamics of the system
    def NeuralDyn(self, input):
        x, u = torch.split(input, split_size_or_sections=1)
        output = self.NeuralEff(u) * u + x
        return output


    def HamiltonianNeural(self, u, x, lam, t):
        input = torch.concat((x, u), dim=0)
        output = self.RunCost(x, u, t) + torch.dot(lam, self.NeuralDyn(input))
        return output

    def Solve(self, sample_num, train_iters, solver='NeuralPMPOffline', PMPlr=1e-3, PMP_iters=3000):

        self.TrainNeuralEfficiency(sample_num=sample_num, train_iters=train_iters)

        if solver == 'NeuralPMPOffline':
            x0 = torch.tensor(self.x0, dtype=torch.float32)
            Solver.NeuralPMPOffline(horizon=self.T, state_dim=self.state_dim, action_dim=self.action_dim, initial=x0,
                                    lr=PMPlr, iters=PMP_iters, action_lower=self.action_lower, action_upper=self.action_upper,
                                    TrueDynamics=self.TrueDynamics, NeuralDynamics=self.NeuralDyn, Terminal=self.Terminal,
                                    HamiltonianNN=self.HamiltonianNeural, Objective=self.Obj
                                    )
        elif solver == 'NeuralPMPOffline_TimeRelated':
            x0 = torch.tensor(self.x0, dtype=torch.float32)
            Solver.NeuralPMPOffline_TimeRelatedCost(horizon=self.T, state_dim=self.state_dim, action_dim=self.action_dim, initial=x0,
                                    lr=PMPlr, iters=PMP_iters, action_lower=self.action_lower, action_upper=self.action_upper,
                                    TrueDynamics=self.TrueDynamics, NeuralDynamics=self.NeuralDyn,
                                    Terminal=self.Terminal,
                                    HamiltonianNN=self.HamiltonianNeural, Objective=self.Obj
                                    )
