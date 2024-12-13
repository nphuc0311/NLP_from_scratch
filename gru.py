import torch
import torch.nn as nn

from typing import overload
from rnn import RNN_Base
from torch import Tensor

from utils import copy_parameters

torch.manual_seed(42)
class GRU(RNN_Base):
    @overload
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False
    ) -> None:
        ...

    def __init__(self, *args, **kwargs):
        mode = "GRU"
        super().__init__(mode, *args, **kwargs)


    def gru_cell(self, x, h_t_minus_1, weights):
        w_ih, w_hh, *biases = weights
        w_ir, w_iz, w_in = w_ih.chunk(3)
        w_hr, w_hz, w_hn = w_hh.chunk(3)

        r_t = x @ w_ir.T + h_t_minus_1 @ w_hr.T
        z_t = x @ w_iz.T + h_t_minus_1 @ w_hz.T
        
        if self.bias:
            b_ih, b_hh = biases
            b_ir, b_iz, b_in = b_ih.chunk(3)
            b_hr, b_hz, b_hn = b_hh.chunk(3)

            r_t += b_ir + b_hr
            z_t += b_iz + b_hz
        
        r_t = torch.sigmoid(r_t)
        z_t = torch.sigmoid(z_t)
        n_t = torch.tanh(x @ w_in.T + b_in + r_t*(h_t_minus_1 @ w_hn.T + b_hn)) if bias \
            else torch.tanh(x @ w_in.T + r_t*(h_t_minus_1 @ w_hn.T))
        h_t = (1 - z_t) * n_t + z_t * h_t_minus_1

        return h_t
    
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
        
        h_t_backward = torch.zeros_like(h_t_forward)

        for layer in range(self.num_layers):
            h_t_minus_1_forward = torch.zeros(batch_size, self.hidden_size) if h_0 is None else h_0
            # _flat_weights = [w_ih_k, w_hh_k, b_ih_k, b_hh_k, 
            #                  (w_ih_k, w_hh_k, b_ih_k, b_hh_k)_reverse]
            index = layer * 2 * 2 * self.num_directions if self.bias else layer * 2 * self.num_directions   # (layer * 2 weights * 2 bias * num_direction )
            weights_forward = [x for x in self._flat_weights[index:index + 4]]
        
            if self.bidirectional:
                h_t_minus_1_backward = torch.zeros_like(h_t_minus_1_forward) if h_0 is None else h_0
                index += 4 if self.bias else 2
                weights_backward = [x for x in self._flat_weights[index:index + 4]]
            
            for t in range(seq_len):
                input = x[t] if layer == 0 else output[t]
                h_t_minus_1_forward = self.gru_cell(input, h_t_minus_1_forward, weights_forward)
                h_t_forward[t] = h_t_minus_1_forward

                if self.bidirectional:
                    input = x[seq_len - t - 1] if layer == 0 else output[seq_len - t - 1]
                    h_t_minus_1_backward = self.gru_cell(input, h_t_minus_1_backward, weights_backward)
                    h_t_backward[seq_len - t - 1] = h_t_minus_1_backward

            output = torch.cat((h_t_forward, h_t_backward), dim=2) if self.bidirectional else h_t_forward
        
        if self.batch_first:
            output = output.transpose(0, 1)
    
        return output


if __name__ == "__main__":
    # Test the RNN from scratch
    input_size = 10
    hidden_size = 20
    num_layers = 5
    batch_size = 3
    seq_len = 5
    bias = False
    bidirectional = False

    pytorch_gru = nn.GRU(input_size, 
                    hidden_size, 
                    num_layers, 
                    bias=bias,
                    batch_first=True, 
                    bidirectional=bidirectional
                )

    gru_from_scratch = GRU(input_size, 
                hidden_size, 
                num_layers, 
                bias=bias,
                batch_first=True, 
                bidirectional=bidirectional)

    input = torch.rand(batch_size, seq_len, input_size)

    copy_parameters(gru_from_scratch, pytorch_gru, num_layers, bias, bidirectional)

    # Test the output again
    out = gru_from_scratch(input)
    pytorch_out, _ = pytorch_gru(input)

    # Compare outputs
    out_dif = torch.abs(pytorch_out - out).max().item()
    assert out_dif < 1e-6, "Your RNN is different with torch.nn.GRU from Pytorch, {}".format(out_dif)
    print("Passed, output difference:", out_dif)