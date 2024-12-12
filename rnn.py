import torch
import torch.nn as nn
import math

from typing import Optional, Tuple, overload
from torch import Tensor
from torch.nn import Parameter
from torch.nn import init

torch.manual_seed(42)

class RNN_Base(nn.Module):
    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        nonlinearity: str = "tanh",
        batch_first: bool = False,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.mode = mode
        if mode == "LSTM":
            gate_size = 4 * hidden_size
        elif mode == "GRU":
            gate_size = 3 * hidden_size
        elif mode == "RNN":
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        if nonlinearity not in ["relu", "tanh"]:
            raise ValueError(
                f"Unknown nonlinearity '{nonlinearity}'. Select from 'tanh' or 'relu'."
            )
        self.nonlinearity = nonlinearity
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        self.num_directions = num_directions
        
        self._flat_weights_names = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = (input_size if layer == 0 else hidden_size * num_directions)            
            
                w_ih = Parameter(torch.empty(gate_size, layer_input_size))
                w_hh = Parameter(torch.empty(gate_size, hidden_size))

                b_ih = Parameter(torch.empty(gate_size))
                # Second bias vector included for CuDNN compatibility. Only one
                # bias vector is needed in standard definition.
                b_hh = Parameter(torch.empty(gate_size))

                layer_params: Tuple[Tensor, ...] = ()
                if bias:
                    layer_params = (w_ih, w_hh, b_ih, b_hh)
                else:
                    layer_params = (w_ih, w_hh)
                            
                suffix = "_reverse" if direction == 1 else ""      
                param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                if bias:
                    param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]

                param_names = [x.format(layer, suffix) for x in param_names]
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._flat_weights_names.extend(param_names)
        
        self._init_flat_weights()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)


    def _init_flat_weights(self):
        self._flat_weights = [
            getattr(self, wn) if hasattr(self, wn) else None
            for wn in self._flat_weights_names
        ]


    def __setattr__(self, attr, value):
        if hasattr(self, "_flat_weights_names") and attr in self._flat_weights_names:
            # keep self._flat_weights up to date if you do self.weight = ...
            idx = self._flat_weights_names.index(attr)
            self._flat_weights[idx] = value
        super().__setattr__(attr, value)



class RNN(RNN_Base):
    @overload
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        nonlinearity: str = "tanh",
        batch_first: bool = False,
        bidirectional: bool = False,
    ) -> None:
        ...

    def __init__(self, *args, **kwargs):
        mode = "RNN"
        super().__init__(mode, *args, **kwargs)

    def rnn_cell(self, x, h_t_minus_1, weights):
        w_ih, w_hh, *biases = weights
        h_t = x @ w_ih.T + h_t_minus_1 @ w_hh.T
        if self.bias:
            b_ih, b_hh = biases
            h_t += b_ih + b_hh
        if self.nonlinearity == "tanh":
            return torch.tanh(h_t)
        else:
            return torch.relu(h_t)
        

    def forward(self, x: Tensor, h_0: Optional[Tensor] = None):
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
                h_t_minus_1_forward = self.rnn_cell(input, h_t_minus_1_forward, weights_forward)
                h_t_forward[t] = h_t_minus_1_forward

                if self.bidirectional:
                    input = x[seq_len - t - 1] if layer == 0 else output[seq_len - t - 1]
                    h_t_minus_1_backward = self.rnn_cell(input, h_t_minus_1_backward, weights_backward)
                    h_t_backward[seq_len - t - 1] = h_t_minus_1_backward

            output = torch.cat((h_t_forward, h_t_backward), dim=2) if bidirectional else h_t_forward
        
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
    bias = True
    bidirectional = False


    num_directions = 2 if bidirectional else 1
    pytorch_rnn = nn.RNN(input_size, 
                    hidden_size, 
                    num_layers, 
                    nonlinearity="relu", 
                    bias=bias,
                    batch_first=True, 
                    bidirectional=bidirectional
                )


    my_rnn = RNN(input_size, 
                hidden_size, 
                num_layers,
                nonlinearity="relu", 
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
    my_out = my_rnn(input)
    pytorch_out, pytorch_ht = pytorch_rnn(input)

    # Compare outputs
    out_dif = torch.abs(pytorch_out - my_out).max().item()
    assert out_dif < 1e-6, "Your RNN is different with torch.nn.RNN from Pytorch, {}".format(out_dif)
    print("Passed, output difference:", out_dif)
