from src.models.crnn import CRNN
from torch import nn
import torch
from torch.autograd import Variable

class CRNN_LSTM(CRNN):
    """
    A mixed deep learning framework with Convolution and LSTM
    """
    def get_code(self):
        return 'CRNN_LSTM'
    
    def __init__(self, feature_num, filters_num, window, ticker_num, hidden_unit_num, hidden_layer_num, dropout_ratio):
        super(CRNN_LSTM, self).__init__()  # Fixed: was super(CRNN, self).__init__()
        self.hidden_layer_num = hidden_layer_num
        self.filters_num = filters_num
        self.hidden_unit_num = hidden_unit_num
        
        # Check if CUDA is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=feature_num * 2,
                      out_channels=filters_num,
                      kernel_size=window,
                      stride=1,
                      padding=0),
            nn.ReLU(True),
        )
        self.bn = nn.BatchNorm1d(filters_num)
        self.pool = nn.MaxPool1d(int(ticker_num*(ticker_num-1)/2))
        self.rnn = nn.LSTM(
            input_size=int(ticker_num*(ticker_num-1)/2),
            hidden_size=hidden_unit_num,
            num_layers=hidden_layer_num,
            dropout=dropout_ratio,
            batch_first=True
        )

        self.line = nn.Linear(filters_num, 1)
        torch.nn.init.xavier_uniform_(self.line.weight)  # Fixed: added underscore
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        h0 = Variable(torch.zeros(self.hidden_layer_num, self.filters_num, self.hidden_unit_num)).to(self.device).float()
        c0 = Variable(torch.zeros(self.hidden_layer_num, self.filters_num, self.hidden_unit_num)).to(self.device).float()
        h0 = torch.nn.init.xavier_uniform_(h0)  # Fixed: added underscore
        c0 = torch.nn.init.xavier_uniform_(c0)  # Fixed: added underscore
        return (h0, c0)  # LSTM