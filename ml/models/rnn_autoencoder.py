import numpy as np
import torch


class AttentionEncoder(torch.nn.Module):
    def __init__(self, input_dim: int,
                 hidden_size: int = 64,
                 num_lstm_layers: int = 1,
                 denoising: bool = False,
                 num_features: int = 11,
                 num_lags: int = 10,
                 matrix_rep: bool = True,
                 architecture: str = "lstm"
                 ):
        super(AttentionEncoder, self).__init__()

        self.hidden_dim = hidden_size
        self.layer_dim = num_lstm_layers
        self.add_noise = denoising
        self.num_features = num_features
        self.num_lags = num_lags
        self.matrix_rep = matrix_rep
        self.architecture = architecture
        if not self.matrix_rep:
            self.input_dim = input_dim // num_lags
        else:
            self.input_dim = input_dim

        if architecture == "lstm":
            self.rnn = torch.nn.LSTM(input_size=self.input_dim, hidden_size=hidden_size, num_layers=1)
        else:
            self.rnn = torch.nn.GRU(input_size=self.input_dim, hidden_size=hidden_size, num_layers=1)
        self.attention = torch.nn.Linear(
            in_features=2 * hidden_size + num_lags,
            out_features=1
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def _get_noise(self, x, sigma=0.01, p=0.1):
        normal = sigma * torch.randn(x.shape)
        mask = np.random.uniform(size=x.shape)
        mask = (mask < p).astype(int)
        noise = normal * torch.tensor(mask)
        return noise

    def forward(self, x, exogenous=None, device="cpu"):
        if len(x.shape) > 2:
            x = x.reshape(x.size(0), x.size(1), x.size(2))
        else:
            x = x.reshape(x.size(0), x.size(1) // self.num_features, x.size(1) // self.num_lags)

        ht = torch.nn.init.xavier_normal_(torch.zeros(1, x.size(0), self.hidden_dim)).to(device)
        ct = torch.nn.init.xavier_normal_(torch.zeros(1, x.size(0), self.hidden_dim)).to(device)

        attention = torch.zeros(x.size(0), self.num_lags, self.input_dim).to(device)
        input_encoded = torch.zeros(x.size(0), self.num_lags, self.hidden_dim).to(device)

        if self.add_noise and self.training:
            x += self._get_noise(x).to(device)

        for t in range(x.size(1)):
            x_ = torch.cat(
                (ht.repeat(self.input_dim, 1, 1).permute(1, 0, 2),
                 ct.repeat(self.input_dim, 1, 1).permute(1, 0, 2),
                 x.permute(0, 2, 1).to(device)), dim=2).to(device)  # bs * input_size * (2 * hidden_dim + seq_len)

            et = self.attention(x_.view(-1, self.hidden_dim * 2 + self.num_lags))  # bs * input_size * 1
            at = self.softmax(et.view(-1, self.input_dim)).to(device)  # (bs * input_size)

            weighted_input = torch.mul(at, x[:, t, :].to(device))  # (bs * input_size)

            self.rnn.flatten_parameters()
            if self.architecture == "lstm":
                _, (ht, ct) = self.rnn(weighted_input.unsqueeze(0), (ht, ct))
            else:
                _, ht = self.rnn(weighted_input.unsqueeze(0), ht)

            input_encoded[:, t, :] = ht
            attention[:, t, :] = at

        return attention, input_encoded


class AttentionDecoder(torch.nn.Module):
    def __init__(self,
                 encoder_hidden_size: int = 64,
                 decoder_hidden_size: int = 64,
                 num_lags: int = 10,
                 out_dim: int = 5,
                 architecture="lstm"
                 ):
        super(AttentionDecoder, self).__init__()

        self.encoder_dim = encoder_hidden_size
        self.decoder_dim = decoder_hidden_size
        self.num_lags = num_lags
        self.architecture = architecture

        self.attention = torch.nn.Sequential(
            torch.nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(encoder_hidden_size, 1)
        )
        if architecture == "lstm":
            self.rnn = torch.nn.LSTM(input_size=out_dim, hidden_size=decoder_hidden_size)
        else:
            self.rnn = torch.nn.GRU(input_size=out_dim, hidden_size=decoder_hidden_size)
        self.fc = torch.nn.Linear(encoder_hidden_size + out_dim, out_dim)
        self.fc_out = torch.nn.Linear(decoder_hidden_size + encoder_hidden_size, out_dim)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc.weight.data.normal_()

    def forward(self, x_encoded, exogenous=None, device="cpu", y_history=None):
        ht = torch.nn.init.xavier_normal_(torch.zeros(1, x_encoded.size(0), self.encoder_dim)).to(device)
        ct = torch.nn.init.xavier_normal_(torch.zeros(1, x_encoded.size(0), self.encoder_dim)).to(device)

        context = torch.autograd.Variable(torch.zeros(x_encoded.size(0), self.encoder_dim))

        for t in range(self.num_lags):
            x = torch.cat((
                ht.repeat(self.num_lags, 1, 1).permute(1, 0, 2),
                ct.repeat(self.num_lags, 1, 1).permute(1, 0, 2),
                x_encoded.to(device)), dim=2)

            x = self.softmax(
                self.attention(
                    x.view(-1, 2 * self.decoder_dim + self.encoder_dim)
                ).view(-1, self.num_lags))

            context = torch.bmm(x.unsqueeze(1), x_encoded.to(device))[:, 0, :]  # bs * encoder_dim

            y_tilde = self.fc(torch.cat((context.to(device), y_history[:, t].to(device)),
                                        dim=1))  # bs * out_dim

            self.rnn.flatten_parameters()
            if self.architecture == "lstm":
                _, (ht, ct) = self.rnn(y_tilde.unsqueeze(0), (ht, ct))
            else:
                _, ht = self.rnn(y_tilde.unsqueeze(0), ht)

        out = self.fc_out(torch.cat((ht[0], context.to(device)), dim=1))  # seq + 1

        return out


class DualAttentionAutoEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, architecture: str = "lstm", matrix_rep: bool = True):
        super(DualAttentionAutoEncoder, self).__init__()
        self.encoder = AttentionEncoder(input_dim, architecture=architecture, matrix_rep=matrix_rep)
        self.decoder = AttentionDecoder(architecture=architecture)

    def forward(self, x, exogenous=None, device="cpu", y_hist=None):
        attentions, encoder_out = self.encoder(x, exogenous, device)
        outputs = self.decoder(encoder_out, exogenous, device, y_hist)
        return outputs
