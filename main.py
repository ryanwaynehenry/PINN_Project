import io
import argparse
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
import scipy.io
from matplotlib import pyplot as plt
import imageio.v2 as imageio

BATCH_SIZE = 256
WORKER_SIZE = 4



class NavierStokes(pl.LightningModule):
    def __init__(self, num_nodes, optim):
        super().__init__()
        self.num_nodes = num_nodes
        self.optim = optim
        self.setup()
        self.act_fun = nn.Tanh()
        self.network = nn.Sequential(
            nn.Linear(3, self.num_nodes), self.act_fun,
            nn.Linear(self.num_nodes, self.num_nodes), self.act_fun,
            nn.Linear(self.num_nodes, self.num_nodes),  self.act_fun,
            nn.Linear(self.num_nodes, self.num_nodes), self.act_fun,
            nn.Linear(self.num_nodes, self.num_nodes), self.act_fun,
            nn.Linear(self.num_nodes, self.num_nodes), self.act_fun,
            nn.Linear(self.num_nodes, self.num_nodes), self.act_fun,
            nn.Linear(self.num_nodes, self.num_nodes), self.act_fun,
            nn.Linear(self.num_nodes, self.num_nodes), self.act_fun,
            nn.Linear(self.num_nodes, 2)
        )

    def forward(self, x, y, t):
        torch.set_grad_enabled(True)
        batch = torch.hstack((x, y, t))
        batch.requires_grad_()
        result = self.network(batch)
        return result

    def loss_calc(self, x, y, t, act_u, act_v, act_pressure):
        result = self.forward(x, y, t)
        psi, pressure = result[:, 0], result[:, 1]
        # u, v , pressure = result[:, 0], result[:, 1], result[:, 2]

        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v = -1.0 * torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
        du_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dy2 = torch.autograd.grad(du_dy, y, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0]
        du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        dv_dx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        dv_dx2 = torch.autograd.grad(dv_dx, x, grad_outputs=torch.ones_like(dv_dx), create_graph=True)[0]
        dv_dy = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        dv_dy2 = torch.autograd.grad(dv_dy, y, grad_outputs=torch.ones_like(dv_dy), create_graph=True)[0]
        dv_dt = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        dpressure_dx = torch.autograd.grad(pressure, x, grad_outputs=torch.ones_like(pressure), create_graph=True)[0]
        dpressure_dy = torch.autograd.grad(pressure, y, grad_outputs=torch.ones_like(pressure), create_graph=True)[0]

        u_loss = self.mse_loss(u, act_u)
        v_loss = self.mse_loss(v, act_v)
        pressure_loss = self.mse_loss(pressure, act_pressure)

        f = du_dt + u * du_dx + v * du_dy + dpressure_dx - 0.01 * (du_dx2 + du_dy2)
        g = dv_dt + u * dv_dx + v * dv_dy + dpressure_dy - 0.01 * (dv_dx2 + dv_dy2)

        f_loss = self.mse_loss(f, self.null)
        g_loss = self.mse_loss(g, self.null)

        traditional_loss = u_loss + v_loss #+ pressure_loss
        physics_loss = f_loss + g_loss
        return traditional_loss + physics_loss

    def training_step(self, batch, batch_idx):
        self.null = self.null.to(self.device)
        features, labels = batch
        x = features[0, :]
        y = features[1, :]
        t = features[2, :]
        x.requires_grad_()
        y.requires_grad_()
        t.requires_grad_()

        u = labels[0, :]
        v = labels[1, :]
        pressure = labels[2, :]

        loss = self.loss_calc(x, y, t, u, v, pressure)
        if self.training_counter % 1 == 0:
            print(f"Iteration {self.training_counter} Train Loss: {loss.item()}")
        self.training_counter += 1
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        torch.inference_mode(False)
        self.train()
        self.null = self.null.to(self.device)

        writer = imageio.get_writer('PressureMap.gif', fps=10)

        features, labels = batch
        u_error_total = 0
        v_error_total = 0
        pressure_error_total = 0
        for i in range(int(len(features[1]) / 5000)):
            print(f"On testing iteration {i}")
            indices = i * np.ones((5000,1)) + (np.arange(0, 5000) * 200).reshape((5000, 1))
            x = features[0, indices].reshape((5000, 1))
            y = features[1, indices].reshape((5000, 1))
            t = features[2, indices].reshape((5000, 1))
            x.requires_grad_()
            y.requires_grad_()
            t.requires_grad_()

            u = labels[0, indices].reshape((5000, 1))
            v = labels[1, indices].reshape((5000, 1))
            pressure = labels[2, indices].reshape((5000, 1))

            result = self.forward(x, y, t)
            # u_out = result[0, :]
            # v_out = result[1, :]
            psi_out = result[:, 0:1]
            pressure_out = result[:, 1:2]

            u_out = torch.autograd.grad(psi_out, y, grad_outputs=torch.ones_like(psi_out), create_graph=True)[0]
            v_out = -1.0 * torch.autograd.grad(psi_out, x, grad_outputs=torch.ones_like(psi_out), create_graph=True)[0]

            u_error, v_error, pressure_error = self.calc_gradient_error(u, v, pressure, u_out, v_out, pressure_out)
            u_error_total += u_error
            v_error_total += v_error
            pressure_error_total += pressure_error


            plot1 = pressure_out.data.cpu().numpy()
            plot2 = pressure.data.cpu().numpy()

            plot1 = np.reshape(plot1, (50, 100))
            plot2 = np.reshape(plot2, (50, 100))

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

            # Plot for output pressure
            c1 = axs[0].contourf(plot1, levels=30, cmap='jet')
            fig.colorbar(c1, ax=axs[0])
            axs[0].set_title('Output Pressure')

            # Plot for true pressure
            c2 = axs[1].contourf(plot2, levels=30, cmap='jet')
            fig.colorbar(c2, ax=axs[1])
            axs[1].set_title('True Pressure')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            # plt.show()
            plt.close(fig)

            buf.seek(0)
            image = imageio.imread(buf)
            writer.append_data(image)

        writer.close()

        print(f"The error with respect to u is {u_error_total}")
        print(f"The error with respect to v is {v_error_total}")
        print(f"The error with respect to pressure is {pressure_error_total}")

        return

    def calc_gradient_error(self, u, v, pressure, u_out, v_out, pressure_out):
        def mean_square_error(gradient1, gradient2):
            return np.mean((gradient1 - gradient2) ** 2)

        def calc_gradient(input):
            return np.array(np.gradient(np.reshape(input.data.cpu().numpy(), (50, 100))))

        u_grad = calc_gradient(u)
        v_grad = calc_gradient(v)
        pressure_grad = calc_gradient(pressure)
        u_out_grad = calc_gradient(u_out)
        v_out_grad = calc_gradient(v_out)
        pressure_out_grad = calc_gradient(pressure_out)

        u_error = mean_square_error(u_out_grad, u_grad)
        v_error = mean_square_error(v_out_grad, v_grad)
        pressure_error = mean_square_error(pressure_out_grad, pressure_grad)

        return u_error, v_error, pressure_error


    def setup(self, stage=None):
        self.N_train = 5000
        self.mse_loss = nn.MSELoss()
        self.null = torch.zeros((self.N_train, 1))
        self.training_counter = 0

        # Load data from MATLAB file
        data = scipy.io.loadmat('cylinder_wake.mat')

        # Extract data from the structure
        U_star = data['U_star']  # N x 2 x T
        P_star = data['p_star']  # N x T
        t_star = data['t']  # T x 1
        X_star = data['X_star']  # N x 2

        # Total number of data points and time instances
        N = X_star.shape[0]
        T = t_star.shape[0]

        # Rearrange data for training
        XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
        YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
        TT = np.tile(t_star, (1, N)).T  # N x T

        UU = U_star[:, 0, :]  # N x T
        VV = U_star[:, 1, :]  # N x T
        PP = P_star  # N x T

        x = torch.tensor(XX.flatten()[:, None], dtype=torch.float32)
        y = torch.tensor(YY.flatten()[:, None], dtype=torch.float32)
        t = torch.tensor(TT.flatten()[:, None], dtype=torch.float32)
        u = torch.tensor(UU.flatten()[:, None], dtype=torch.float32)
        v = torch.tensor(VV.flatten()[:, None], dtype=torch.float32)
        pressure = torch.tensor(PP.flatten()[:, None], dtype=torch.float32)

        # Randomly select indices for training data
        idx = np.random.choice(N * T, self.N_train, replace=False)
        x_train = x[idx]
        y_train = y[idx]
        t_train = t[idx]
        u_train = u[idx]
        v_train = v[idx]
        p_train = pressure[idx]

        # Create datasets
        self.train_dataset = CustomDataset((x_train, y_train, t_train), (u_train, v_train, p_train))
        self.test_dataset = CustomDataset((x, y, t), (u, v, pressure))

    # Create DataLoaders
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.N_train, shuffle=False, num_workers=WORKER_SIZE,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.N_train * 200, shuffle=False, num_workers=WORKER_SIZE,
                          persistent_workers=True)


    def configure_optimizers(self):
        if self.optim == "LBFGS":
            optimizer = torch.optim.LBFGS(self.parameters(), lr=1, max_iter=10000, max_eval=10000,
                                      history_size=50, tolerance_grad=1e-05, tolerance_change=0.5 * np.finfo(float).eps,
                                      line_search_fn="strong_wolfe")
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_model(model, load_model, save_model, load_filename, save_filename, train_flag, optim):

    logger = TensorBoardLogger("tb_logs", name="my_model")
    if optim == "LBFGS":
        max_epochs = 1
    else:
        max_epochs = 500

    if torch.cuda.is_available():
    # if False:
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator='gpu', logger=logger, inference_mode=False)
    else:
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator='cpu', logger=logger, inference_mode=False)
    if load_model:
        model.network.load_state_dict(torch.load(load_filename))
    if train_flag:
        trainer.fit(model)
    if save_model:
        torch.save(model.network.state_dict(), save_filename)
    return model, trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Run Navier-Stokes model training and testing.")
    parser.add_argument('--load_file', action='store_true', help='Flag to load a pre-trained model.')
    parser.add_argument('--save_file', action='store_true', help='Flag to load a pre-trained model.')
    parser.add_argument('--load_filename', type=str, default='model.pth', help='Filename of the model to load.')
    parser.add_argument('--save_filename', type=str, default='model.pth', help='Filename of the model to load.')
    parser.add_argument('--train',  action='store_true', help='Flag to train the model.')
    parser.add_argument('--num_nodes', type=int, default=30, help='Number of nodes per hidden layer in the model.')
    parser.add_argument('--optimization', type=str, default='LBFGS', help='Type of optimizer to use (e.g., adam, LBFGS).')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load the model here (make sure to define how to load inside your model or training utility)
    if args.load_file:
        load_model = True
    else:
        load_model = False
    if args.save_file:
        save_model = True
    else:
        save_model = False
    load_filename = args.load_filename
    save_filename = args.save_filename
    print(args.train)
    train_flag = args.train
    print(train_flag)
    num_nodes = args.num_nodes
    optim = args.optimization

    torch.set_float32_matmul_precision('high')
    model = NavierStokes(num_nodes, optim)
    model, trainer = train_model(model, load_model, save_model, load_filename, save_filename, train_flag, optim)

    trainer.test(model)



