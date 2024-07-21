import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from pipeline.models import freqency
#from ...pipeline.models.freqency import SimpleMultiLayerModel
from pipeline.utils import RunningAverageMeter


def get_data(device=torch.device('cpu')):
    x = torch.linspace(-np.pi, np.pi, 1000).reshape(-1, 1)
    y_1 = torch.sin(x)
    y_2 = torch.sin(4 * x)
    y = y_1 + y_2
    return x.to(device), y.to(device), y_1.to(device), y_2.to(device)


def get_continous_freqs_data(device=torch.device('cpu')):
    N = 384
    x = torch.linspace(-10, 10, N).reshape(-1, 1)
    duration = 20
    #y = torch.from_numpy((np.array(range(N)) / duration).astype(np.int32) % 3 -1).to(torch.float)
    y = torch.ones_like(x)
    i = 2
    while (i < N):
        y[i:min(max(i, i + 1 * duration), N)] = 0
        y[min(max(i, i + 1 * duration), N):min(max(i, i + 3 * duration), N)] = -1
        y[min(max(i, i + 3 * duration), N):min(max(i, i + 4 * duration), N)] = 0
        y[min(max(i, i + 4 * duration), N):min(max(i, i + 6 * duration), N)] = 1
        i += 6 * duration

    return x.to(device), y.to(device)


def visualize_subplot(ax, x, y, x_label='x', y_label='y', color='red'):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot(x, y, color=color)


def visualize_line(ax, x, y, color='red'):
    ax.plot(x, y, color=color)


def visualize_fft(ax, x, y, color='red'):
    N = x.shape[0]
    T = x[-1] - x[0]
    y = np.reshape(y, [N])
    fft_w = np.fft.fft(y)
    freqs = np.fft.fftfreq(N, T / N) * 4
    left_freqs = int(N / 2)
    maglitude = np.log10(np.abs(fft_w) + 1e-1)
    ax.plot(freqs[:left_freqs], maglitude[:left_freqs], color)
    #visualize_line(ax, freqs, maglitude, color)


def visualize(x, y, pred_y, iter):

    fig = plt.figure(figsize=(30, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    visualize_subplot(ax1, x, y)
    #visualize_line(ax1, x, np.sin(x), color = 'blue')
    visualize_line(ax1, x, pred_y, color='green')
    ax2 = fig.add_subplot(1, 2, 2)
    visualize_fft(ax2, x, y)
    visualize_fft(ax2, x, pred_y, color='green')
    fig.tight_layout()
    plt.savefig('./data/images/{:04d}'.format(iter))
    plt.close()


def high_low_freqency_test():
    device = torch.device('cuda')
    lr = 5e-5
    max_steps = 6000
    visualize_steps = 100
    print_steps = 100
    loss_meter = RunningAverageMeter()
    model = freqency.SimpleMultiLayerModel(1, 256, 1).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr)
    mse_loss = torch.nn.MSELoss()
    total_loss_list = []
    y1_loss_list = []
    y2_loss_list = []
    for iter in range(max_steps):
        opt.zero_grad()
        batch_x, batch_y, y_1, y_2 = get_data(device=device)
        pred_y = model(batch_x)
        loss = mse_loss(pred_y, batch_y)
        loss.backward()
        opt.step()
        loss_meter.update(loss.item())
        y1_loss = torch.mean(torch.abs(pred_y - y_1)**2)
        y2_loss = torch.mean(torch.abs(pred_y - y_2)**2)
        total_loss_list.append(loss.item())
        y1_loss_list.append(y1_loss.item())
        y2_loss_list.append(y2_loss.item())
        if iter % print_steps == 0:
            print('Iter {:04d} | Total Loss {:.6f},y_1 Loss {:.6f},y_2 Loss {:.6f}'.format(iter, loss.item(), y1_loss.item(), y2_loss.item()))
        if iter % visualize_steps == 0:
            visualize(batch_x.detach().cpu().numpy(), batch_y.detach().cpu().numpy(), pred_y.detach().cpu().numpy(), iter)
    x_steps = list(range(max_steps))
    fig = plt.figure(figsize=(30, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    visualize_subplot(ax1, x_steps, total_loss_list)
    visualize_subplot(ax1, x_steps, y1_loss_list, color='green')
    visualize_subplot(ax1, x_steps, y2_loss_list, color='blue')
    fig.tight_layout()
    plt.savefig('./data/images/loss.png')
    plt.close()


def continous_freqency_test():
    device = torch.device('cpu')
    lr = 1e-3
    decay_num = 2
    visualize_epochs = 1000
    print_epochs = 1000
    batch_size = 128
    max_epochs = 40000
    decay_epochs = (max_epochs * (1 - 1 / np.power(decay_num, np.arange(decay_num) + 1))).astype(np.uint)
    #print(decay_epochs)
    loss_meter = RunningAverageMeter()
    model = freqency.SimpleMultiLayerModel(1, 256, 1).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr)
    step_lr = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=decay_epochs.tolist(), gamma=0.1)
    mse_loss = torch.nn.MSELoss()
    data_x, data_y = get_continous_freqs_data(device=device)
    data_size = data_x.shape[0]
    for epoch in range(max_epochs):
        for iter in range(data_size // batch_size):
            opt.zero_grad()
            mask = np.random.choice(data_size, batch_size, replace=False)
            batch_x = data_x[mask]
            batch_y = data_y[mask]
            pred_y = model(batch_x)
            loss = mse_loss(pred_y, batch_y)
            loss.backward()
            opt.step()
            loss_meter.update(loss.item())
        step_lr.step()
        if epoch % print_epochs == 0:
            print('Epoch {:04d}| Total Loss {:.6f}, lr {:.6f}'.format(epoch, loss_meter.val, step_lr.get_last_lr()[0]))
        if epoch % visualize_epochs == 0:
            data_pred_y = model(data_x)
            visualize(data_x.detach().cpu().numpy(), data_y.detach().cpu().numpy(), data_pred_y.detach().cpu().numpy(), epoch)


__all__ = ["Frequency"]


class Frequency:

    def __init__(self, args, config):
        self.args = args
        self.config = config

    def run(self):
        high_low_freqency_test()
