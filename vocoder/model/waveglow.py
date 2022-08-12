import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from vocoder.hparams import hparams as hps



class WaveGlowLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(WaveGlowLoss, self).__init__()
        self.sigma = sigma

    def forward(self, z, log_s_list, log_det_W_list):
        # for i, log_s in enumerate(log_s_list):
        #     if i == 0:
        #         log_s_total = torch.sum(log_s)
        #         log_det_W_total = log_det_W_list[i]
        #         print(log_s_total + log_det_W_total)
        #     else:
        #         log_s_total = log_s_total + torch.sum(log_s)
        #         log_det_W_total += log_det_W_list[i]

        # 0807 https://github.com/yoyololicon/constant-memory-waveglow/ 참조 로스 계산 식 변경
        for i, log_s_t in enumerate(log_s_list):
            if i == 0:
                log_det_W_log_s_total = log_det_W_list[i] + torch.sum(log_s_t)

            else:
                log_det_W_log_s_total += log_det_W_list[i] + torch.sum(log_s_t)

        # loss = torch.sum(z * z) / (2 * self.sigma * self.sigma) - log_s_total - log_det_W_total
        loss = torch.sum(z * z) / (2 * self.sigma * self.sigma) - log_det_W_log_s_total
        # return loss / (z.size(0) * z.size(1) * z.size(2))
        return loss / (z.size(0) * z.size(1) * z.size(2)), torch.sum(z * z).item(), (z.size(0) * z.size(1) * z.size(2)), log_det_W_log_s_total.item()



@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b

    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])

    acts = t_act * s_act
    return acts



class Invertible1x1Conv(nn.Module):
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0, bias=False)

        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]

        W = W.view(c, c, 1)

        self.conv.weight.data = W


    def forward(self, z, reverse=False):
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()

        if reverse:
            if not hasattr(self, 'W_inverse'):
                W_inverse = W.inverse()
                W_inverse = Variable(W_inverse[..., None])

                # if z.type() == 'torch.cuda.HalfTensor':
                #     W_inverse = W_inverse.half()

                self.W_inverse = W_inverse

            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z

        else:
            log_det_W = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W



class WN(nn.Module):
    def __init__(self, n_in_channels):
        super(WN, self).__init__()
        self.n_in_channels = n_in_channels
        self.n_mel_channels = hps.num_mels * hps.n_group
        self.n_layers = hps.wn_n_layers
        self.n_channels = hps.wn_n_channels
        self.n_kernel_size = hps.wn_kernel_size

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.cond_layers = nn.ModuleList()

        start = nn.Conv1d(self.n_in_channels, self.n_channels, 1)
        start = nn.utils.weight_norm(start, name='weight')
        self.start = start

        end = nn.Conv1d(self.n_channels, 2 * self.n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        # cond_layer = nn.Conv1d(self.n_mel_channels, 2 * self.n_channels * self.n_layers, 1)
        # self.cond_layer = nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(self.n_layers):
            dilation = 2 ** i
            padding = int((self.n_kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(self.n_channels, 2 * self.n_channels, self.n_kernel_size, dilation=dilation, padding=padding)
            in_layer = nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            cond_layer = nn.Conv1d(self.n_mel_channels, 2 * self.n_channels, 1)
            cond_layer = nn.utils.weight_norm(cond_layer, name='weight')
            self.cond_layers.append(cond_layer)

            if i < self.n_layers - 1:
                res_skip_channels = 2 * self.n_channels

            else:
                res_skip_channels = self.n_channels

            res_skip_layer = nn.Conv1d(self.n_channels, res_skip_channels, 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)


    def forward(self, audio, spect):
        audio = self.start(audio)

        for i in range(self.n_layers):
            acts = fused_add_tanh_sigmoid_multiply(self.in_layers[i](audio), self.cond_layers[i](spect), torch.IntTensor([self.n_channels]))

            res_skip_acts = self.res_skip_layers[i](acts)

            if i < self.n_layers - 1:
                audio = res_skip_acts[:, :self.n_channels, :] + audio
                skip_acts = res_skip_acts[:, self.n_channels:, :]

            else:
                skip_acts = res_skip_acts

            if i == 0:
                output = skip_acts

            else:
                output = skip_acts + output

        return self.end(output)

        # output = torch.zeros_like(audio)
        # n_channels_tensor = torch.IntTensor([self.n_channels])

        # spect = self.cond_layer(spect)

        # for i in range(self.n_layers):
        #     spect_offset = i * 2 * self.n_channels

        #     acts = fused_add_tanh_sigmoid_multiply(self.in_layers[i](audio),
        #                                             spect[:, spect_offset:spect_offset + 2 * self.n_channels, :],
        #                                             n_channels_tensor)

        #     res_skip_acts = self.res_skip_layers[i](acts)

        #     if i < self.n_layers - 1:
        #         audio = audio + res_skip_acts[:, :self.n_channels, :]
        #         output = output + res_skip_acts[:, :self.n_channels, :]

        #     else:
        #         output = output + res_skip_acts

        # return self.end(output)





class WaveGlow(nn.Module):
    def __init__(self):
        super(WaveGlow, self).__init__()
        # hparams
        self.n_flows = hps.n_flows
        self.n_group = hps.n_group
        self.n_early_every  = hps.n_early_every
        self.n_early_size  = hps.n_early_size

        self.upsample = nn.ConvTranspose1d(hps.num_mels, hps.num_mels, hps.upsample_kerner_size, stride=hps.upsample_stride)

        self.WN = nn.ModuleList()
        self.convinv = nn.ModuleList()

        n_half = int(self.n_group / 2)
        n_remaining_channels = self.n_group

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size/2)
                n_remaining_channels = n_remaining_channels - self.n_early_size

            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.WN.append(WN(n_half))

        self.n_remaining_channels = n_remaining_channels


    def forward(self, spect, audio):
        """
        waveglow_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        waveglow_input[1] = audio: batch x time
        """
        # spect, audio = waveglow_input

        spect = self.upsample(spect)

        assert(spect.size(2) >= audio.size(1))

        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)

        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)

        output_audio = []
        log_s_list = []
        log_det_W_list = []

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, :self.n_early_size, :])
                audio = audio[:, self.n_early_size:, :]

            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN[k](audio_0, spect)
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]

            audio_1 = torch.exp(log_s) * audio_1 + b
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list


    def infer(self, spect, sigma=1.0):
        spect = self.upsample(spect)

        trim_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-trim_cutoff]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)

        # if spect.type() == 'torch.cuda.HalfTensor':
        #     audio = torch.cuda.HalfTensor(spect.size(0),
        #                                   self.n_remaining_channels,
        #                                   spect.size(2)).normal_()

        # else:
        #     audio = torch.cuda.FloatTensor(spect.size(0),
        #                                    self.n_remaining_channels,
        #                                    spect.size(2)).normal_()

        if hps.is_cuda:
            audio = torch.cuda.FloatTensor(spect.size(0), self.n_remaining_channels, spect.size(2)).normal_()

        else:
            audio = torch.FloatTensor(spect.size(0), self.n_remaining_channels, spect.size(2)).normal_()

        audio = Variable(sigma * audio)

        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1) / 2)

            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN[k](audio_0, spect)

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.convinv[k](audio, reverse=True)

            if k % self.n_early_every == 0 and k > 0:
                # if spect.type() == 'torch.cuda.HalfTensor':
                #     z = torch.cuda.HalfTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                # else:
                #     z = torch.cuda.FloatTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()

                if hps.is_cuda:
                    z = torch.cuda.FloatTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()

                else:
                    z = torch.FloatTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()

                audio = torch.cat((sigma * z, audio), 1)

        audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1).data
        return audio


    # @staticmethod
    # def remove_weightnorm(model):
    #     waveglow = model
    #     for WN in waveglow.WN:
    #         WN.start = torch.nn.utils.remove_weight_norm(WN.start)
    #         WN.in_layers = remove(WN.in_layers)
    #         WN.cond_layer = torch.nn.utils.remove_weight_norm(WN.cond_layer)
    #         WN.res_skip_layers = remove(WN.res_skip_layers)
    #     return waveglow
    @staticmethod
    def remove_weightnorm(model):
        waveglow = model
        for WN in waveglow.WN:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layers = remove(WN.cond_layers)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveglow



def remove(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list