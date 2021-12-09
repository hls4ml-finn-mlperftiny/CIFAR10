import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense
from qkeras.qlayers import QDense, QActivation
from qkeras.qconvolutional import QConv2D
from qkeras.qpooling import QAveragePooling2D
from qkeras.quantizers import quantized_bits, quantized_relu


def count_nonzero_weights(model):
    nonzero = total = 0
    layer_count_alive = {}
    layer_count_total = {}
    layer_names = [layer.name for layer in model.layers]
    for name in layer_names:
        module = model.get_layer(name)
        if isinstance(module, Conv2D) or isinstance(module, QConv2D) or isinstance(module, Dense) or isinstance(module, QDense):
            p_list = module.get_weights()
            for idx, p in enumerate(p_list):
                if idx == 0:
                    curr_name = name + ".weight"
                elif idx == 1:
                    curr_name = name + ".bias"
                tensor = p
                nz_count = np.count_nonzero(tensor)
                total_params = np.prod(tensor.shape)
                layer_count_alive.update({curr_name: nz_count})
                layer_count_total.update({curr_name: total_params})
                nonzero += nz_count
                total += total_params
                print(f'{curr_name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    return nonzero, total, layer_count_alive, layer_count_total


def calc_BOPS(model, input_data_precision=32):
    last_bit_width = input_data_precision
    alive, total, l_alive, l_total = count_nonzero_weights(model)
    #b_w = model.weight_precision if hasattr(model, 'weight_precision') else 32
    total_BOPS = 0
    for module in model.layers:
        if isinstance(module, Conv2D) or isinstance(module, QConv2D) or isinstance(module, Dense) or isinstance(module, QDense):
            b_a = last_bit_width  # TODO: If have time fix this to have activations bit width (bit width of incoming)
            b_w = int(module.get_quantization_config()['kernel_quantizer'][15]) + 1  # plus 1 to account for sign bit qkeras ignores
            n = np.prod(module.input.shape[1:])
            m = np.prod(module.output.shape[1:])
            if isinstance(module, Conv2D) or isinstance(module, QConv2D):
                k = np.prod(module.kernel_size)
            else:
                k = 1
            total = l_total[module.name + '.weight'] + l_total[module.name + '.bias']
            alive = l_alive[module.name + '.weight'] + l_alive[module.name + '.bias']
            p = 1 - ((total - alive) / total)  # fraction of layer remaining
            # assuming b_a is the output bitwidth of the last layer
            #module_BOPS = m*n*p*(b_a*b_w + b_a + b_w + math.log2(n))
            module_BOPS = m * n * k * k * (p * b_a * b_w + b_a + b_w + math.log2(n*k*k))
            print("{} BOPS: {} = {}*{}*{}({}*{}*{} + {} + {} + {})".format(module.name, module_BOPS, m, n, k*k, p, b_a, b_w, b_a, b_w, math.log2(n*k*k)))
            last_bit_width = b_w  # TODO: If have time fix this to have activations bit width (bit width of incoming)
            total_BOPS += module_BOPS
            # TODO: If have time add skip connections
    scientific_notation = "{:.2e}".format(total_BOPS)
    print("Total BOPS: {} = {}".format(total_BOPS, scientific_notation))

    return total_BOPS
