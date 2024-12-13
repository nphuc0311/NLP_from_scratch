import torch
import torch.nn as nn

def copy_parameters(model_1, model_2, num_layers, bias, bidirectional):
    num_directions = 2 if bidirectional else 1

    # Copy parameters from model_2 to model_1 for checking result
    with torch.no_grad():
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = "_reverse" if direction == 1 else ""
                
                # Parameter names in model_2
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
                
                # Parameter names in model_1
                my_weight_ih_name = f"weight_ih_l{layer}{suffix}"
                my_weight_hh_name = f"weight_hh_l{layer}{suffix}"
                if bias:
                    my_bias_ih_name = f"bias_ih_l{layer}{suffix}"
                    my_bias_hh_name = f"bias_hh_l{layer}{suffix}"
                
                # Assign parameters
                setattr(model_1, my_weight_ih_name, nn.Parameter(getattr(model_2, weight_ih_name).detach().clone()))
                setattr(model_1, my_weight_hh_name, nn.Parameter(getattr(model_2, weight_hh_name).detach().clone()))
                if bias:
                    setattr(model_1, my_bias_ih_name, nn.Parameter(getattr(model_2, bias_ih_name).detach().clone()))
                    setattr(model_1, my_bias_hh_name, nn.Parameter(getattr(model_2, bias_hh_name).detach().clone()))