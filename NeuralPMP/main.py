import argparse
from Env import Env


# Create an instance of the ArgumentParser class
parser = argparse.ArgumentParser()

# Add arguments to the parser
parser.add_argument('--Env_Name', type=str, help='Enter your environment', default='LQR')
parser.add_argument('--Solver', type=str, help='Choose your solver for optimal control', default='NeuralPMPOffline')
parser.add_argument('--Sample_Num', type=int, help='Set sample number of Dynamics training', default=2000)
parser.add_argument('--Training_iters', type=int, help='Set iterations of Dynamics training', default=50000)
parser.add_argument('--PMPlr', type=float, help='Set learning rate for Neural-PMP', default=0.001)
parser.add_argument('--PMP_iters', type=int, help='Set iterations for Neural-PMP', default=15000)


# Parse the command-line arguments
args = parser.parse_args()


if __name__ == '__main__':
    Env_name = args.Env_Name

    if Env_name == 'LQR':
        Env_Selected = Env.LQR()
    elif Env_name == 'Battery':
        Env_Selected = Env.Battery()

    else:
        Env_Selected = Env.LQR()
        print('Env not Supported! Will automatically select LQR for optimizing.')

    Solver = args.Solver
    sample_num = args.Sample_Num
    train_iters = args.Training_iters
    PMPlr = args.PMPlr
    PMPiters = args.PMP_iters


    Env_Selected.Solve(sample_num=sample_num, train_iters=train_iters, solver=Solver, PMPlr=PMPlr, PMP_iters=PMPiters)