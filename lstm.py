import torch
import torch.nn as nn

from typing import overload
from rnn import RNN_Base
from torch import Tensor

torch.manual_seed(42)
class LSTM(RNN_Base):
    @overload
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0
    ) -> None:
        ...

    def __init__(self, *args, **kwargs):
        mode = "LSTM"
        super().__init__(mode, *args, **kwargs)


    def lstm_cell(self, x, h_t_minus_1, c_t_minus_1, weights):
        w_ih, w_hh, *biases = weights
        w_ii, w_if, w_ig, w_io = w_ih.chunk(4)
        w_hi, w_hf, w_hg, w_ho = w_hh.chunk(4)

        i_t = x @ w_ii.T + h_t_minus_1 @ w_hi.T
        f_t = x @ w_if.T + h_t_minus_1 @ w_hf.T
        g_t = x @ w_ig.T + h_t_minus_1 @ w_hg.T
        o_t = x @ w_io.T + h_t_minus_1 @ w_ho.T

        if self.bias:
            b_ih, b_hh = biases
            b_ii, b_if, b_ig, b_io = b_ih.chunk(4)
            b_hi, b_hf, b_hg, b_ho = b_hh.chunk(4)
            
            i_t += b_ii + b_hi
            f_t += b_if + b_hf
            g_t += b_ig + b_hg
            o_t += b_io + b_ho

        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        o_t = torch.sigmoid(o_t)
        g_t = torch.tanh(g_t)

        c_t = f_t * c_t_minus_1 + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
    
    def forward(self, x: Tensor, h_0: Tensor = None, c_0: Tensor = None):
        if self.batch_first:
            x = x.transpose(0, 1)  # Transpose to (seq_len, batch, input_size)

        seq_len, batch_size, _ = x.size()
        h_t_forward = torch.zeros(
                        seq_len,
                        batch_size,
                        self.hidden_size,
                        device=x.device,
                        dtype=x.dtype
                    )
        
        c_t_forward = torch.zeros_like(h_t_forward)
        h_t_backward = torch.zeros_like(h_t_forward)
        c_t_backward = torch.zeros_like(h_t_forward)

        for layer in range(self.num_layers):
            h_t_minus_1_forward = torch.zeros(batch_size, self.hidden_size) if h_0 is None else h_0
            c_t_minus_1_forward =  torch.zeros_like(h_t_minus_1_forward) if c_0 is None else c_0
            # _flat_weights = [w_ih_k, w_hh_k, b_ih_k, b_hh_k, 
            #                  (w_ih_k, w_hh_k, b_ih_k, b_hh_k)_reverse]
            index = layer * 2 * 2 * self.num_directions if self.bias else layer * 2 * self.num_directions   # (layer * 2 weights * 2 bias * num_direction )
            weights_forward = [x for x in self._flat_weights[index:index + 4]]
        
            if self.bidirectional:
                h_t_minus_1_backward = torch.zeros_like(h_t_minus_1_forward) if h_0 is None else h_0
                c_t_minus_1_backward =  torch.zeros_like(h_t_minus_1_forward) if c_0 is None else c_0
                index += 4 if self.bias else 2
                weights_backward = [x for x in self._flat_weights[index:index + 4]]
            
            for t in range(seq_len):
                input = x[t] if layer == 0 else output[t]
                h_t_minus_1_forward, c_t_minus_1_forward = self.lstm_cell(input, h_t_minus_1_forward, c_t_minus_1_forward, weights=weights_forward)
                h_t_forward[t], c_t_forward[t] = h_t_minus_1_forward, c_t_minus_1_forward

                if self.bidirectional:
                    input = x[seq_len - t - 1] if layer == 0 else output[seq_len - t - 1]
                    h_t_minus_1_backward, c_t_minus_1_backward = self.lstm_cell(input, h_t_minus_1_backward, c_t_minus_1_backward, weights_backward)
                    h_t_backward[seq_len - t - 1], c_t_backward[seq_len - t - 1] = h_t_minus_1_backward, c_t_minus_1_backward

            output = torch.cat((h_t_forward, h_t_backward), dim=2) if self.bidirectional else h_t_forward
            c_t = torch.cat((c_t_forward, c_t_backward), dim=2) if self.bidirectional else c_t_forward
        
        if self.batch_first:
            output = output.transpose(0, 1)
    
        return output, c_t


if __name__ == "__main__":
    # Test the RNN from scratch
    input_size = 10
    hidden_size = 20
    num_layers = 5
    batch_size = 3
    seq_len = 5
    bias = True
    bidirectional = True


    num_directions = 2 if bidirectional else 1
    pytorch_rnn = nn.LSTM(input_size, 
                    hidden_size, 
                    num_layers, 
                    bias=bias,
                    batch_first=True, 
                    bidirectional=bidirectional
                )


    my_rnn = LSTM(input_size, 
                hidden_size, 
                num_layers, 
                bias=bias,
                batch_first=True, 
                bidirectional=bidirectional)

    input = torch.rand(batch_size, seq_len, input_size)

    # Copy parameters from pytorch_rnn to my_rnn for checking result
    with torch.no_grad():
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = "_reverse" if direction == 1 else ""
                
                # Parameter names in pytorch_rnn
                weight_ih_name = f"weight_ih_l{layer}"
                weight_hh_name = f"weight_hh_l{layer}"
                if bias:
                    bias_ih_name = f"bias_ih_l{layer}"
                    bias_hh_name = f"bias_hh_l{layer}"
                
                if direction == 1:  # Reverse for bidirectional
                    weight_ih_name += "_reverse"
                    weight_hh_name += "_reverse"
                    if bias:
                        bias_ih_name += "_reverse"
                        bias_hh_name += "_reverse"
                
                # Parameter names in my_rnn
                my_weight_ih_name = f"weight_ih_l{layer}{suffix}"
                my_weight_hh_name = f"weight_hh_l{layer}{suffix}"
                if bias:
                    my_bias_ih_name = f"bias_ih_l{layer}{suffix}"
                    my_bias_hh_name = f"bias_hh_l{layer}{suffix}"
                
                # Assign parameters
                setattr(my_rnn, my_weight_ih_name, nn.Parameter(getattr(pytorch_rnn, weight_ih_name).detach().clone()))
                setattr(my_rnn, my_weight_hh_name, nn.Parameter(getattr(pytorch_rnn, weight_hh_name).detach().clone()))
                if bias:
                    setattr(my_rnn, my_bias_ih_name, nn.Parameter(getattr(pytorch_rnn, bias_ih_name).detach().clone()))
                    setattr(my_rnn, my_bias_hh_name, nn.Parameter(getattr(pytorch_rnn, bias_hh_name).detach().clone()))

    # Test the output again
    my_out, my_c_out = my_rnn(input)
    pytorch_out, pytorch_ht = pytorch_rnn(input)

    # Compare outputs
    out_dif = torch.abs(pytorch_out - my_out).max().item()
    assert out_dif < 1e-6, "Your RNN is different with torch.nn.LSTM from Pytorch, {}".format(out_dif)
    print("Passed, output difference:", out_dif)