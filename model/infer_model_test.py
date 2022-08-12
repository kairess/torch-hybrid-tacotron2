import torch
from torch import nn
from math import sqrt
from hparams import hparams as hps
from torch.autograd import Variable
from torch.nn import functional as F
from model.layers import ConvNorm, LinearNorm
from utils.util import to_var, get_mask_from_lengths
from utils.custom_util import guided_attentions


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets, iteration):
        ## model_output : [mel_outputs, mel_outputs_postnet, linear_outputs, gate_outputs, alignments]
        ## targets : [mel_padded, linear_padded, gate_padded]
        mel_target, linear_target, gate_target = targets[0], targets[1], targets[2]
        mel_target.requires_grad = False
        linear_target.requires_grad = False
        gate_target.requires_grad = False
        slice = torch.arange(0, gate_target.size(1), hps.n_frames_per_step)
        gate_target = gate_target[:, slice].view(-1, 1)

        mel_out, mel_out_postnet, linear_out, gate_out, alignments = model_output

        if hps.use_guided_attention:
            input_lengths, target_lengths = targets[3], targets[4]
            input_lengths.requires_grad = False
            target_lengths.requires_grad = False

            alignments_mask = guided_attentions(input_lengths, target_lengths, alignments.size(1),
                                                g=hps.guided_attention_sigma, use_eos=hps.use_eos)

            alignments_mask[:, :, 0] = 0

            attention_loss = torch.mean(alignments * alignments_mask)
            attention_loss_item = attention_loss.item()

        gate_out = gate_out.view(-1, 1)
        p = hps.p
        mel_loss = nn.MSELoss()(p * mel_out, p * mel_target) + \
                   nn.MSELoss()(p * mel_out_postnet, p * mel_target)
        linear_loss = nn.MSELoss()(p * linear_out, p * linear_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        mel_loss_item = (mel_loss / (p ** 2)).item()
        linear_loss_item = (linear_loss / (p ** 2)).item()
        gate_loss_item = gate_loss.item()

        if hps.use_guided_attention:
            return mel_loss + linear_loss + gate_loss + attention_loss, (mel_loss / (p ** 2) + linear_loss / (
                        p ** 2) + gate_loss + attention_loss).item(), mel_loss_item, linear_loss_item, gate_loss_item, attention_loss_item

        else:
            return mel_loss + linear_loss + gate_loss, (mel_loss / (p ** 2) + linear_loss / (
                        p ** 2) + gate_loss).item(), mel_loss_item, linear_loss_item, gate_loss_item


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float('inf')

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        '''
		PARAMS
		------
		query: decoder output (batch, num_mels * n_frames_per_step)
		processed_memory: processed encoder outputs (B, T_in, attention_dim)
		attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

		RETURNS
		-------
		alignment (batch, max_time)
		'''

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def apply_windowing(self, alignment, forcing_idx):
        win_back = 2
        win_front = 6
        back_win = forcing_idx - win_back
        front_win = forcing_idx + win_front

        if back_win > 0:
            alignment[:, :back_win] = self.score_mask_value

        if front_win < alignment.size(1):
            alignment[:, front_win:] = self.score_mask_value

        if forcing_idx == 0:
            alignment[:, 0] = alignment.max()

        return alignment


    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        '''
		PARAMS
		------
		attention_hidden_state: attention rnn last output
		memory: encoder outputs
		processed_memory: processed encoder outputs
		attention_weights_cat: previous and cummulative attention weights
		mask: binary mask for padded data
		'''
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights

    def inference(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, forcing_sequence, mask):
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        if forcing_sequence is not None:
            # prev_argmax = torch.argmax(alignment, dim=1).item()
            # if prev_argmax >= alignment.size(1) - 5 and forcing_sequence != alignment.size(1) - 1:
            #     alignment[:, forcing_sequence - 1] *= 1.5
            #     alignment[:, forcing_sequence + 1] *= 1.5
            #
            # alignment[:, forcing_sequence] *= 2.0
            alignment = self.apply_windowing(alignment, forcing_sequence)
            alignment[:, forcing_sequence] *= 2.0

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    '''Postnet
		- Five 1-d convolution with 512 channels and kernel size 5
	'''

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hps.num_mels, hps.postnet_embedding_dim,
                         kernel_size=hps.postnet_kernel_size, stride=1,
                         padding=int((hps.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hps.postnet_embedding_dim))
        )

        for i in range(1, hps.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hps.postnet_embedding_dim,
                             hps.postnet_embedding_dim,
                             kernel_size=hps.postnet_kernel_size, stride=1,
                             padding=int((hps.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hps.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hps.postnet_embedding_dim, hps.num_mels,
                         kernel_size=hps.postnet_kernel_size, stride=1,
                         padding=int((hps.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hps.num_mels))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


## for Linear spectrogram
class BatchNormConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding,
                 activation=None):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_dim, out_dim,
                                kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)


class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class CBHG(nn.Module):
    """CBHG module: a recurrent neural network composed of:
        - 1-d convolution banks
        - Highway networks + residual connections
        - Bidirectional gated recurrent units
    """

    def __init__(self, in_dim, K=16, projections=[128, 128]):
        super(CBHG, self).__init__()
        self.in_dim = in_dim
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList(
            [BatchNormConv1d(in_dim, in_dim, kernel_size=k, stride=1,
                             padding=k // 2, activation=self.relu)
             for k in range(1, K + 1)])
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        in_sizes = [K * in_dim] + projections[:-1]
        activations = [self.relu] * (len(projections) - 1) + [None]
        self.conv1d_projections = nn.ModuleList(
            [BatchNormConv1d(in_size, out_size, kernel_size=3, stride=1,
                             padding=1, activation=ac)
             for (in_size, out_size, ac) in zip(
                in_sizes, projections, activations)])

        self.pre_highway = nn.Linear(projections[-1], in_dim, bias=False)
        self.highways = nn.ModuleList(
            [Highway(in_dim, in_dim) for _ in range(4)])

        self.gru = nn.GRU(
            in_dim, in_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        # (B, T_in, in_dim)
        x = inputs

        # Needed to perform conv1d on time-axis
        # (B, in_dim, T_in)
        if x.size(-1) == self.in_dim:
            x = x.transpose(1, 2)

        T = x.size(-1)

        # (B, in_dim*K, T_in)
        # Concat conv1d bank outputs
        x = torch.cat([conv1d(x)[:, :, :T] for conv1d in self.conv1d_banks], dim=1)
        assert x.size(1) == self.in_dim * len(self.conv1d_banks)
        x = self.max_pool1d(x)[:, :, :T]

        for conv1d in self.conv1d_projections:
            x = conv1d(x)

        # (B, T_in, in_dim)
        # Back to the original shape
        x = x.transpose(1, 2)

        if x.size(-1) != self.in_dim:
            x = self.pre_highway(x)

        # Residual connection
        x += inputs
        for highway in self.highways:
            x = highway(x)

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True)

        # (B, T_in, in_dim*2)
        outputs, _ = self.gru(x)

        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)

        return outputs


## for Linear spectrogram


class Encoder(nn.Module):
    '''Encoder module:
		- Three 1-d convolution banks
		- Bidirectional LSTM
	'''

    def __init__(self):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hps.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hps.encoder_embedding_dim,
                         hps.encoder_embedding_dim,
                         kernel_size=hps.encoder_kernel_size, stride=1,
                         padding=int((hps.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hps.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hps.encoder_embedding_dim,
                            int(hps.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.num_mels = hps.num_mels
        self.n_frames_per_step = hps.n_frames_per_step
        self.encoder_embedding_dim = hps.encoder_embedding_dim
        self.attention_rnn_dim = hps.attention_rnn_dim
        self.decoder_rnn_dim = hps.decoder_rnn_dim
        self.prenet_dim = hps.prenet_dim
        self.max_decoder_steps = hps.max_decoder_steps
        self.gate_threshold = hps.gate_threshold
        self.p_attention_dropout = hps.p_attention_dropout
        self.p_decoder_dropout = hps.p_decoder_dropout

        self.prenet = Prenet(
            hps.num_mels * hps.n_frames_per_step,
            [hps.prenet_dim, hps.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hps.prenet_dim + hps.encoder_embedding_dim,
            hps.attention_rnn_dim)

        self.attention_layer = Attention(
            hps.attention_rnn_dim, hps.encoder_embedding_dim,
            hps.attention_dim, hps.attention_location_n_filters,
            hps.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hps.attention_rnn_dim + hps.encoder_embedding_dim,
            hps.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hps.decoder_rnn_dim + hps.encoder_embedding_dim,
            hps.num_mels * hps.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hps.decoder_rnn_dim + hps.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        ''' Gets all zeros frames to use as first decoder input
		PARAMS
		------
		memory: decoder outputs

		RETURNS
		-------
		decoder_input: all zeros frames
		'''
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.num_mels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        ''' Initializes attention rnn states, decoder rnn states, attention
		weights, attention cumulative weights, attention context, stores memory
		and stores processed memory
		PARAMS
		------
		memory: Encoder outputs
		mask: Mask for padded data if training, expects None for inference
		'''
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        ''' Prepares decoder inputs, i.e. mel outputs
		PARAMS
		------
		decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

		RETURNS
		-------
		inputs: processed decoder inputs

		'''
        # (B, num_mels, T_out) -> (B, T_out, num_mels)
        decoder_inputs = decoder_inputs.transpose(1, 2).contiguous()
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step), -1)
        # (B, T_out, num_mels) -> (T_out, B, num_mels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        ''' Prepares decoder outputs for output
		PARAMS
		------
		mel_outputs:
		gate_outputs: gate output energies
		alignments:

		RETURNS
		-------
		mel_outputs:
		gate_outpust: gate output energies
		alignments:
		'''
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, num_mels) -> (B, T_out, num_mels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.num_mels)
        # (B, T_out, num_mels) -> (B, num_mels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        ''' Decoder step using stored states, attention and memory
		PARAMS
		------
		decoder_input: previous mel output

		RETURNS
		-------
		mel_output:
		gate_output: gate output energies
		attention_weights:
		'''
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights


    def decode_infer(self, decoder_input , decoder_hidden, decoder_cell, 
                        attention_hidden, attention_cell, attention_weights,
                        attention_weights_cum, attention_context, 
                        memory, processed_memory, forcing_sequence):

        cell_input = torch.cat((decoder_input, attention_context), -1)
        attention_hidden, attention_cell = self.attention_rnn(
            cell_input, (attention_hidden, attention_cell))
        attention_hidden = F.dropout(attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat((attention_weights.unsqueeze(1), attention_weights_cum.unsqueeze(1)), dim=1)
        attention_context, attention_weights = self.attention_layer.inference(attention_hidden, memory, processed_memory, 
                                                                    attention_weights_cat, forcing_sequence, None)
        attention_weights_cum += attention_weights
        decoder_input = torch.cat((attention_hidden, attention_context), -1)
        decoder_hidden, decoder_cell = self.decoder_rnn(decoder_input, (decoder_hidden, decoder_cell))
        decoder_hidden = F.dropout(decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat((decoder_hidden, attention_context), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return decoder_output, gate_prediction, attention_weights, (attention_hidden, attention_cell, decoder_hidden, decoder_cell, attention_weights, attention_weights_cum, attention_context)

    def get_initailize_decoder_states(self, memory, mask):
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        processed_memory = self.attention_layer.memory_layer(memory)
        mask = mask

        return attention_hidden, attention_cell, decoder_hidden, decoder_cell, attention_weights, attention_weights_cum, attention_context, processed_memory, mask


    def forward(self, memory, decoder_inputs, memory_lengths):
        ''' Decoder forward pass for training
		PARAMS
		------
		memory: Encoder outputs
		decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
		memory_lengths: Encoder output lengths for attention masking.

		RETURNS
		-------
		mel_outputs: mel outputs from the decoder
		gate_outputs: gate outputs from the decoder
		alignments: sequence of attention weights from the decoder
		'''
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory, text_sequence, postnet_layer):
        ''' Decoder inference
		PARAMS
		------
		memory: Encoder outputs

		RETURNS
		-------
		mel_outputs: mel outputs from the decoder
		gate_outputs: gate outputs from the decoder
		alignments: sequence of attention weights from the decoder
		'''
        decoder_input = self.get_go_frame(memory)

        # self.initialize_decoder_states(memory, mask=None)
        states = self.get_initailize_decoder_states(memory, mask=None)
        attention_hidden = states[0]
        attention_cell = states[1]
        decoder_hidden = states[2]
        decoder_cell = states[3]
        attention_weights = states[4]
        attention_weights_cum = states[5]
        attention_context = states[6]
        processed_memory = states[7]

        mel_outputs, gate_outputs, alignments = [], [], []

        prev_argmax = 0
        duration = []
        repeat_states = states
        repeat_decoder_input = decoder_input
        forcing_sequence = None

        move_step = True
        t = 0
        max_seq_len = memory.size(1)
        max_window = 8

        from text import sequence_to_text
        from text.custom_symbols import _JAMO_LEADS ,_JAMO_TAILS, _JAMO_VOWELS
        from utils.util import to_arr
        import numpy as np
        text_sequence = to_arr(text_sequence)[0].astype(np.int16).tolist()
        text_sequence = sequence_to_text(text_sequence) + '~'
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment, prev_states = self.decode_infer(decoder_input, decoder_hidden, decoder_cell,
                                                             attention_hidden, attention_cell, attention_weights,
                                                             attention_weights_cum, attention_context,
                                                             memory, processed_memory, forcing_sequence)

            # states : attention_hidden, attention_cell, decoder_hidden, decoder_cell, attention_weights, attention_weights_cum, attention_context, processed_memory, mask
            # prev_states : attention_hidden, attention_cell, decoder_hidden, decoder_cell, attention_weights, attention_weights_cum, attention_context
            forcing_sequence = torch.argmax(alignment, dim=1).item()
            # print('timp : ', t, 'argmax : ', forcing_sequence)

            # if prev_argmax == forcing_sequence or prev_argmax + 1 == forcing_sequence:
            # if prev_argmax == forcing_sequence or prev_argmax <= forcing_sequence:
            if prev_argmax <= forcing_sequence and forcing_sequence <= prev_argmax + 2:
                duration += [forcing_sequence]

                attention_hidden = prev_states[0]
                attention_cell = prev_states[1]
                decoder_hidden = prev_states[2]
                decoder_cell = prev_states[3]
                attention_weights = prev_states[4]
                attention_weights_cum = prev_states[5]
                attention_context = prev_states[6]

                postnet_input = mel_output.view(-1, self.num_mels).unsqueeze(0).transpose(1, 2).contiguous()
                mel_output_postnet = postnet_layer(postnet_input).squeeze(0)
                mel_output_postnet = mel_output_postnet.view(1, -1)
                mel_output = mel_output + mel_output_postnet

                mel_outputs += [mel_output.squeeze(1)]
                gate_outputs += [gate_output]
                alignments += [alignment]

                decoder_input = mel_output

                repeat_states = prev_states
                repeat_decoder_input = mel_output

                # make_mute = False
                # if text_sequence[forcing_sequence] in [',', '.', ' ', '?', '!']:
                #     empty_size = 2 if text_sequence[forcing_sequence] in [',', ' '] else 3
                #     max_window += 1
                #     for _ in range(empty_size):
                #         decoder_input = self.prenet(decoder_input)
                #         mel_output, gate_output, alignment, prev_states = self.decode_infer(decoder_input, decoder_hidden, decoder_cell,
                #                                                          attention_hidden, attention_cell, attention_weights,
                #                                                          attention_weights_cum, attention_context,
                #                                                          memory, processed_memory, forcing_sequence)

                #         forcing_sequence = torch.argmax(alignment, dim=1).item()

                #         duration += [forcing_sequence]

                #         attention_hidden = prev_states[0]
                #         attention_cell = prev_states[1]
                #         decoder_hidden = prev_states[2]
                #         decoder_cell = prev_states[3]
                #         attention_weights = prev_states[4]
                #         attention_weights_cum = prev_states[5]
                #         attention_context = prev_states[6]

                #         mel_outputs += [mel_output.squeeze(1)]
                #         gate_outputs += [gate_output]
                #         alignments += [alignment]

                #         decoder_input = mel_output
                #         repeat_states = prev_states
                #         repeat_decoder_input = mel_output

                #     if len(duration[-max_window:]) == max_window and all([prev_argmax == argmax for argmax in duration[-max_window:]]):
                #         make_mute = False

                #     else:
                #         make_mute = True


                # if text_sequence[forcing_sequence] in _JAMO_TAILS or text_sequence[forcing_sequence] in _JAMO_VOWELS:
                #     # print(text_sequence[forcing_sequence])
                #     max_window = 3

                # else:
                #     max_window = 4

                # if text_sequence[forcing_sequence] in _JAMO_LEADS:
                #     max_window = 4

                # elif text_sequence[forcing_sequence] in _JAMO_VOWELS:
                #     max_window = 5
                
                # else:
                #     max_window = 8

                max_window = 8
                if len(duration[-max_window:]) == max_window and all([prev_argmax == argmax for argmax in duration[-max_window:]]):
                    next_forcing_sequence = prev_argmax + 1 if prev_argmax < max_seq_len - 1 else None
                    if all([max_seq_len - 1 == argmax for argmax in duration[-max_window:]]):
                        print('Terminated by duration.')
                        break

                    # print('increase forcing sequence')

                else:
                    next_forcing_sequence = None

                prev_argmax = forcing_sequence
                forcing_sequence = next_forcing_sequence



            else:
                attention_hidden = repeat_states[0]
                attention_cell = repeat_states[1]
                decoder_hidden = repeat_states[2]
                decoder_cell = repeat_states[3]
                attention_weights = repeat_states[4]
                attention_weights_cum = repeat_states[5]
                attention_context = repeat_states[6]

                decoder_input = repeat_decoder_input

                forcing_sequence = prev_argmax


            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                if max_seq_len - 1 == min(duration[-5:]):
                    print('Terminated by gate.')
                    break
                # elif len(mel_outputs) > 1 and is_end_of_frames(mel_output):
                #     print('Warning: End with low power.')
                #     break
            elif len(mel_outputs) == self.max_decoder_steps * 2:
                print('Warning: Reached max decoder steps.')
                break

            elif t == self.max_decoder_steps * 2:
                print('Warning: Reached max decoder steps.')
                break

            t += 1

        # for _ in range(max_window):
        #     forcing_sequence = 0
        #     decoder_input = self.prenet(decoder_input)
        #     mel_output, gate_output, alignment, prev_states = self.decode_infer(decoder_input, decoder_hidden, decoder_cell,
        #                                                      attention_hidden, attention_cell, attention_weights,
        #                                                      attention_weights_cum, attention_context,
        #                                                      memory, processed_memory, forcing_sequence)

        #     forcing_sequence = torch.argmax(alignment, dim=1).item()

        #     duration += [forcing_sequence]

        #     attention_hidden = prev_states[0]
        #     attention_cell = prev_states[1]
        #     decoder_hidden = prev_states[2]
        #     decoder_cell = prev_states[3]
        #     attention_weights = prev_states[4]
        #     attention_weights_cum = prev_states[5]
        #     attention_context = prev_states[6]

        #     mel_outputs += [mel_output.squeeze(1)]
        #     gate_outputs += [gate_output]
        #     alignments += [alignment]

        #     decoder_input = mel_output


        print(duration)
        print([text_sequence[i] for i in duration])

        new_mel_outputs = []

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

    # def inference(self, memory, text_sequence):
    #     ''' Decoder inference
    #     PARAMS
    #     ------
    #     memory: Encoder outputs
    #
    #     RETURNS
    #     -------
    #     mel_outputs: mel outputs from the decoder
    #     gate_outputs: gate outputs from the decoder
    #     alignments: sequence of attention weights from the decoder
    #     '''
    #     decoder_input = self.get_go_frame(memory)
    #
    #     # self.initialize_decoder_states(memory, mask=None)
    #     states = self.get_initailize_decoder_states(memory, mask=None)
    #     attention_hidden = states[0]
    #     attention_cell = states[1]
    #     decoder_hidden = states[2]
    #     decoder_cell = states[3]
    #     attention_weights = states[4]
    #     attention_weights_cum = states[5]
    #     attention_context = states[6]
    #     processed_memory = states[7]
    #
    #     mel_outputs, gate_outputs, alignments = [], [], []
    #
    #     duration = []
    #     forcing_sequence = 0
    #
    #     t = 0
    #     max_seq_len = memory.size(1)
    #
    #     from text import sequence_to_text
    #     from text.custom_symbols import _JAMO_LEADS, _JAMO_TAILS, _JAMO_VOWELS
    #     from utils.util import to_arr
    #     import numpy as np
    #     text_sequence = to_arr(text_sequence)[0].astype(np.int16).tolist()
    #     text_sequence = sequence_to_text(text_sequence) + '~'
    #     while True:
    #         jamo = text_sequence[forcing_sequence]
    #
    #         if jamo in _JAMO_LEADS:
    #             loop_cnt = 2
    #
    #         elif jamo in _JAMO_VOWELS:
    #             loop_cnt = 4
    #
    #         elif jamo in _JAMO_TAILS:
    #             loop_cnt = 4
    #
    #         else:
    #             loop_cnt = 8
    #
    #         for _ in range(loop_cnt):
    #             decoder_input = self.prenet(decoder_input)
    #             mel_output, gate_output, alignment, prev_states = self.decode_infer(decoder_input, decoder_hidden,
    #                                                                                 decoder_cell,
    #                                                                                 attention_hidden, attention_cell,
    #                                                                                 attention_weights,
    #                                                                                 attention_weights_cum,
    #                                                                                 attention_context,
    #                                                                                 memory, processed_memory,
    #                                                                                 forcing_sequence)
    #
    #             forcing_sequence = torch.argmax(alignment, dim=1).item()
    #
    #             duration += [forcing_sequence]
    #
    #             attention_hidden = prev_states[0]
    #             attention_cell = prev_states[1]
    #             decoder_hidden = prev_states[2]
    #             decoder_cell = prev_states[3]
    #             attention_weights = prev_states[4]
    #             attention_weights_cum = prev_states[5]
    #             attention_context = prev_states[6]
    #
    #             mel_outputs += [mel_output.squeeze(1)]
    #             gate_outputs += [gate_output]
    #             alignments += [alignment]
    #
    #             decoder_input = mel_output
    #
    #         forcing_sequence += 1
    #
    #         if forcing_sequence == max_seq_len:
    #             break
    #
    #         if torch.sigmoid(gate_output.data) > self.gate_threshold:
    #             if max_seq_len - 1 == min(duration[-5:]):
    #                 print('Terminated by gate.')
    #                 break
    #
    #         elif len(mel_outputs) == self.max_decoder_steps * 2:
    #             print('Warning: Reached max decoder steps.')
    #             break
    #
    #         elif t == self.max_decoder_steps * 2:
    #             print('Warning: Reached max decoder steps.')
    #             break
    #
    #         t += 1
    #
    #     print(duration)
    #     print([text_sequence[i] for i in duration])
    #
    #     # new_mel_outputs = []
    #     #
    #     # for i in range(len(duration)):
    #     #     jamo = text_sequence[i]
    #     #
    #     #     if jamo in _JAMO_LEADS:
    #     #         new_mel_outputs += [mel_outputs[i] * 3]
    #     #
    #     #     elif jamo in _JAMO_LEADS:
    #     #         new_mel_outputs += [mel_outputs[i] * 5]
    #     #
    #     #     elif jamo in _JAMO_LEADS:
    #     #         new_mel_outputs += [mel_outputs[i] * 5]
    #     #
    #     #     else:
    #     #         new_mel_outputs += [mel_outputs[i] * 8]
    #     #
    #     # mel_outputs = new_mel_outputs
    #
    #     # for i in range(max_seq_len):
    #     #     if i not in duration:
    #     #         print(i, text_sequence[i], text_sequence[i - 5:i + 5])
    #
    #     mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
    #         mel_outputs, gate_outputs, alignments)
    #     return mel_outputs, gate_outputs, alignments


def is_end_of_frames(output, eps=0.2):
    return (output.data <= eps).all()


class Tacotron2(nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()
        self.num_mels = hps.num_mels
        self.num_linears = hps.n_fft // 2 + 1
        self.mask_padding = hps.mask_padding
        self.n_frames_per_step = hps.n_frames_per_step
        self.embedding = nn.Embedding(
            hps.n_symbols, hps.symbols_embedding_dim)
        std = sqrt(2.0 / (hps.n_symbols + hps.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = Postnet()
        self.cbhg = CBHG(self.num_mels, K=hps.cbhg_kernel_size, projections=hps.cbhg_projections)
        self.dens_layer = nn.Linear(self.num_mels * 2, self.num_linears)  # 80 * 2, 1025

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, linear_padded, gate_padded, output_lengths = batch
        text_padded = to_var(text_padded).long()
        input_lengths = to_var(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_var(mel_padded).float()
        linear_padded = to_var(linear_padded).float()
        gate_padded = to_var(gate_padded).float()
        output_lengths = to_var(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, linear_padded, max_len, output_lengths),
            (mel_padded, linear_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        ## outputs : [mel_outputs, mel_outputs_postnet, linear_outputs, gate_outputs, alignments]
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths, True)  # (B, T)
            mask = mask.expand(self.num_mels, mask.size(0), mask.size(1))  # (80, B, T)
            mask = mask.permute(1, 0, 2)  # (B, 80, T)

            linear_mask = ~get_mask_from_lengths(output_lengths, True)
            linear_mask = linear_mask.expand(self.num_linears, linear_mask.size(0), linear_mask.size(1))  # (1025, B, T)
            linear_mask = linear_mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)  # (B, 80, T)
            outputs[1].data.masked_fill_(mask, 0.0)  # (B, 80, T)
            outputs[2].data.masked_fill_(linear_mask, 0.0)
            slice = torch.arange(0, mask.size(2), self.n_frames_per_step)
            outputs[3].data.masked_fill_(mask[:, 0, slice], 1e3)  # gate energies (B, T//n_frames_per_step)

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, linears, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        cbhg_outputs = self.cbhg(mel_outputs_postnet.transpose(1, 2))

        linear_outputs = self.dens_layer(cbhg_outputs)
        linear_outputs = linear_outputs.transpose(1, 2)

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, linear_outputs, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs, inputs, self.postnet)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        cbhg_outputs = self.cbhg(mel_outputs_postnet.transpose(1, 2))

        linear_outputs = self.dens_layer(cbhg_outputs)
        linear_outputs = linear_outputs.transpose(1, 2)

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, linear_outputs, gate_outputs, alignments])

        return outputs

    def teacher_infer(self, inputs, mels):
        il, _ = torch.sort(torch.LongTensor([len(x) for x in inputs]),
                           dim=0, descending=True)
        text_lengths = to_var(il)

        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        cbhg_outputs = self.cbhg(mel_outputs_postnet.transpose(1, 2))

        linear_outputs = self.dens_layer(cbhg_outputs)
        linear_outputs = linear_outputs.transpose(1, 2)

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, linear_outputs, gate_outputs, alignments])
