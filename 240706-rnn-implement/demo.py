import torch
import torch.nn as nn

# RNN 分类器，使用 PyTorch 内置 RNN 模块
class RnnClassifier(nn.Module):
    def __init__(self, in_c, hid_c, n_layer=2):
        super(RnnClassifier, self).__init__()
        self.rnn = nn.RNN(input_size=in_c, hidden_size=hid_c, num_layers=n_layer)
        self.classifier = nn.Sequential(
            nn.Linear(hid_c, 1),
            nn.Sigmoid()
        )

    def forward(self, x, hn=None):
        # x: (T, B, in_c) - 序列长度 T, 批次大小 B, 输入特征数 in_c
        # hn: (n_layer, B, hid_c) - 隐藏状态
        x, hn = self.rnn(x, hn)
        x = self.classifier(x[-1, :, :])  # 使用最后一个时间步的输出进行分类
        return x, hn.detach()

# 自定义 RNN 分类器，手动实现 2 层 RNN
class MyRnnClassifier(nn.Module):
    def __init__(self, in_c, hid_c):
        super(MyRnnClassifier, self).__init__()
        self.hid_c = hid_c
        # 第 0 层权重
        self.Wih0 = nn.Linear(in_c, hid_c)  # 输入到隐藏
        self.Whh0 = nn.Linear(hid_c, hid_c)  # 隐藏到隐藏
        # 第 1 层权重
        self.Wih1 = nn.Linear(hid_c, hid_c)  # 输入到隐藏（来自第 0 层输出）
        self.Whh1 = nn.Linear(hid_c, hid_c)  # 隐藏到隐藏

        self.classifier = nn.Sequential(
            nn.Linear(hid_c, 1),
            nn.Sigmoid()
        )
        self.tanh = nn.Tanh()

    def forward(self, x, hn=None):
        # x: (T, B, in_c)
        # hn: (2, B, hid_c)
        T, B, _ = x.shape
        if hn is None:
            hn = torch.zeros(2, B, self.hid_c, dtype=x.dtype, device=x.device)
        hn0, hn1 = hn[0], hn[1]  # 解包两层隐藏状态
        for t in range(T):
            xt = x[t]  # 当前时间步输入 (B, in_c)
            hn0 = self.tanh(self.Wih0(xt) + self.Whh0(hn0))  # 第 0 层更新
            hn1 = self.tanh(self.Wih1(hn0) + self.Whh1(hn1))  # 第 1 层更新
        x = self.classifier(hn1)  # 基于第 1 层最终隐藏状态分类
        hn = torch.stack([hn0, hn1], dim=0).detach()  # 堆叠并分离
        return x, hn

# LSTM 分类器，使用 PyTorch 内置 LSTM 模块
class LstmClassifier(nn.Module):
    def __init__(self, in_c, hid_c, n_layer=1):
        super(LstmClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=in_c, hidden_size=hid_c, num_layers=n_layer)
        self.classifier = nn.Sequential(
            nn.Linear(hid_c, 1),
            nn.Sigmoid()
        )

    def forward(self, x, hn=None):
        # x: (T, B, in_c)
        # hn: (n_layer, B, hid_c) - 隐藏状态和细胞状态
        x, hn = self.lstm(x, hn)
        x = self.classifier(x[-1, :, :])  # 使用最后一个时间步的输出
        return x, hn

# 自定义 LSTM 分类器，手动实现单层 LSTM
class MyLstmClassifier(nn.Module):
    def __init__(self, in_c, hid_c):
        super(MyLstmClassifier, self).__init__()
        self.hid_c = hid_c
        # 输入门权重
        self.Wii = nn.Linear(in_c, hid_c)
        self.Whi = nn.Linear(hid_c, hid_c)
        # 遗忘门权重
        self.Wif = nn.Linear(in_c, hid_c)
        self.Whf = nn.Linear(hid_c, hid_c)
        # 细胞门权重
        self.Wig = nn.Linear(in_c, hid_c)
        self.Whg = nn.Linear(hid_c, hid_c)
        # 输出门权重
        self.Wio = nn.Linear(in_c, hid_c)
        self.Who = nn.Linear(hid_c, hid_c)

        self.classifier = nn.Sequential(
            nn.Linear(hid_c, 1),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, state=None):
        # x: (T, B, in_c)
        # state: ((B, hid_c), (B, hid_c)) - 隐藏状态和细胞状态
        T, B, _ = x.shape
        if state is None:
            ht = torch.zeros(B, self.hid_c, dtype=x.dtype, device=x.device)
            ct = torch.zeros(B, self.hid_c, dtype=x.dtype, device=x.device)
        else:
            ht, ct = state
        for t in range(T):
            xt = x[t]  # (B, in_c)
            # 计算门
            it = self.sigmoid(self.Wii(xt) + self.Whi(ht))  # 输入门
            ft = self.sigmoid(self.Wif(xt) + self.Whf(ht))  # 遗忘门
            gt = self.tanh(self.Wig(xt) + self.Whg(ht))     # 细胞门
            ot = self.sigmoid(self.Wio(xt) + self.Who(ht))  # 输出门
            # 更新细胞和隐藏状态
            ct = ft * ct + it * gt
            ht = ot * self.tanh(ct)
        x = self.classifier(ht)
        return x, (ht.detach(), ct.detach())

# 拷贝 RNN 参数从内置到自定义实现
def copy_params_rnn(_from, _to):
    _dict = {}
    for k, v in _from.state_dict().items():
        if 'classifier' not in k:
            wei, name, num = k.split('.')[1].split('_')
            k_new = f'W{name}{num[1]}.{wei}'
            _dict[k_new] = v
        else:
            _dict[k] = v
    _to.load_state_dict(_dict)
    return _to

# 拷贝 LSTM 参数从内置到自定义实现
def copy_params_lstm(_from, _to):
    _dict = {}
    for k, v in _from.state_dict().items():
        if '_ih_' in k or '_hh_' in k:
            hid_each = v.shape[0] // 4
            wei, name, _ = k.split('.')[1].split('_')
            _dict[f'W{name[0]}i.{wei}'] = v[:hid_each]
            _dict[f'W{name[0]}f.{wei}'] = v[hid_each : hid_each*2]
            _dict[f'W{name[0]}g.{wei}'] = v[hid_each * 2 : hid_each * 3]
            _dict[f'W{name[0]}o.{wei}'] = v[hid_each * 3 :]
        else:
            _dict[k] = v
    _to.load_state_dict(_dict)
    return _to

if __name__ == '__main__':
    x = torch.randn(5, 2, 4)  # (T, B, in_c) - 序列长度 5, 批次大小 2, 输入特征 4
    print('------------ testing rnn ------------')
    Rnn = RnnClassifier(in_c=4, hid_c=8)
    MyRnn = MyRnnClassifier(in_c=4, hid_c=8)
    MyRnn = copy_params_rnn(Rnn, MyRnn)
    y1, state1 = Rnn(x)
    y2, state2 = MyRnn(x)
    print(y1 == y2)  # 检查输出是否相等
    print(state1, state2)  # 打印状态

    print('------------ testing lstm ------------')
    Lstm = LstmClassifier(in_c=4, hid_c=8)
    MyLstm = MyLstmClassifier(in_c=4, hid_c=8)
    MyLstm = copy_params_lstm(Lstm, MyLstm)
    y1, state1 = Lstm(x)
    y2, state2 = MyLstm(x)
    print(y1 == y2)  # 检查输出是否相等
    print(state1, state2)  # 打印状态