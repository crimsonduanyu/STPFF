# -*- coding: utf-8 -*-            
# @Time : 2022/10/16 12:45
# @Author: 段钰
# @EMAIL： duanyu@bjtu.edu.cn
# @FileName: LSTM.py
# @Software: PyCharm
import scipy.io as sio
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
from torch.autograd import Variable
from math import floor
from tqdm import tqdm


#  Define LSTM
class LSTM(nn.Module):
    """
            Parameters：
            - input_size: feature size
            - hidden_size: number of hidden units
            - output_size: number of output
            - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear1 = nn.Linear(hidden_size, output_size)
        pass

    def forward(self, _x):
        # _x = _x.to(device)
        x, _ = self.lstm(_x)  # _x is input, size(seq_len, batch, input_size)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        return x

    pass


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("Training on GPU")
    else:
        device = torch.device("cpu")

    data = pd.read_csv('data\\statMatrix.csv')
    data = data.drop(columns='Unnamed: 0')

    data_x = np.array(data).astype('float32')
    data_y = np.array(data).astype('float32')

    data_len = len(data_x)
    t = np.linspace(0, data_len, data_len)

    train_data_ratio = 0.8
    train_data_len = floor(data_len * train_data_ratio)

    train_x = data_x[:train_data_len]
    train_y = data_y[:train_data_len]
    t_for_training = t[:train_data_len]

    test_x = data_x[train_data_len:]
    test_y = data_y[train_data_len:]
    t_for_testing = t[train_data_len:]

    #   ------------------- training -------------------------
    INPUT_FEATURES_NUM = 8
    OUTPUT_FEATURES_NUM = 8
    train_x_tensor = train_x.reshape(-1, 1, INPUT_FEATURES_NUM)  # set batchsize to 1
    train_y_tensor = train_y.reshape(-1, 1, OUTPUT_FEATURES_NUM)  # set batchsize to 1

    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_tensor = torch.from_numpy(train_y_tensor)

    lstm_model = LSTM(INPUT_FEATURES_NUM, 20, output_size=OUTPUT_FEATURES_NUM, num_layers=3)
    lstm_model = lstm_model.to(device)
    print('LSTM Model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)
    print('train X tensor dimension:', Variable(train_x_tensor).size())

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

    prev_loss = 10
    max_epoch = 1000

    train_x_tensor = train_x_tensor.to(device)
    train_y_tensor = train_y_tensor.to(device)
    with tqdm(total=max_epoch) as bar:
        for epoch in range(max_epoch):
            bar.update(1)
            output = lstm_model(train_x_tensor).to(device)
            loss = criterion(output, train_y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #torch.save(lstm_model.state_dict(),'log\epoch{},Loss:{}.pth'.format(epoch + 1, loss.item()))  # save model parameters to files
            if loss < prev_loss:
                torch.save(lstm_model.state_dict(), 'lstm_model.pth')  # save model parameters to files
                prev_loss = loss

            if loss.item() < 1e-4:
                print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epoch, loss.item()))
                print("The loss value is reached")
                break
            elif (epoch + 1) % 50 == 0:
                print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epoch, loss.item()))

    # prediction on training dataset
    pred_y_for_train = lstm_model(train_x_tensor).to(device)
    pred_y_for_train = pred_y_for_train.view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()

    # ----------------------- test -----------------------
    lstm_model = lstm_model.eval()  # switching to testing moddel

    # prediction on test dataset
    test_x_tensor = test_x.reshape(-1, 1, INPUT_FEATURES_NUM)
    test_x_tensor = torch.from_numpy(test_x_tensor)
    test_x_tensor = test_x_tensor.to(device)

    pred_y_for_test = lstm_model(test_x_tensor).to(device)
    pred_y_for_test = pred_y_for_test.view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()

    loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(test_y))
    print("Test Loss:", loss.item())

    # --------------------- plot -------------------------
    plt.figure()
    plt.plot(t_for_training, train_y[:, 0], 'b', label='y_train')
    plt.plot(t_for_training, pred_y_for_train[:, 0], 'y--', label='pre_train')

    plt.plot(t_for_testing, test_y[:, 0], 'k', label='y_test')
    plt.plot(t_for_testing, pred_y_for_test[:, 0], 'm--', label='pre_test')

    plt.xlabel('t')
    plt.ylabel('Vce')
    plt.show()
