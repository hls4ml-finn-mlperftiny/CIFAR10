import os
# edit depending on where Vivado is installed:
# os.environ['PATH'] = '/<Xilinx installation directory>/Vivado/<version>/bin:' + os.environ['PATH']
os.environ['PATH'] = '/opt/local/Xilinx/Vivado/2019.1/bin:' + os.environ['PATH']
import tensorflow as tf
from qkeras.utils import _add_supported_quantized_objects
import hls4ml
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu

from train import yaml_load

import argparse


def print_dict(d, indent=0):
    align=20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))
            
def main(args):
    # parameters
    config = yaml_load(args.config)
    save_dir = config['save_dir']
    model_name = config['model']['name']
    model_file_path = os.path.join(save_dir, 'model_best.h5')


    from tensorflow.keras.models import load_model
    from qkeras.utils import _add_supported_quantized_objects
    co = {}
    _add_supported_quantized_objects(co)
    

    model = load_model(model_file_path, custom_objects=co)
    print(model.summary)


    import hls4ml
    hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
    hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
    hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model'] = {}
    config['Model']['ReuseFactor'] = 10
    config['Model']['Strategy'] = 'Resource'
    config['Model']['Precision'] = 'ap_fixed<6,4>'
    for name in config['LayerName'].keys():
        config['LayerName'][name]['ReuseFactor'] = 10
        config['LayerName'][name]['Precision'] = 'ap_fixed<6,4>'
    config['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<18,8>'
    config['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,4>'
    config['LayerName']['softmax']['Strategy'] = 'Stable'

    cfg = hls4ml.converters.create_backend_config(fpga_part='xc7z020clg400-1')
    cfg['HLSConfig'] = config
    cfg['IOType'] = 'io_stream'
    cfg['Backend'] = 'Pynq'
    cfg['Interface'] = 's_axilite' # or 'm_axi'
    cfg['KerasModel'] = model
    cfg['OutputDir'] = 'my-hls-test-tiny'

    print("-----------------------------------")
    print_dict(cfg)
    print("-----------------------------------")


    # Bitfile time 
    cfg['OutputDir'] = 'my-hls-test-tiny'
    hls_model = hls4ml.converters.keras_to_hls(cfg)
    hls_model.compile()
    hls_model.build(csim=False,synth=True,export=True)
    hls4ml.report.read_vivado_report('my-hls-test-tiny/')
    hls4ml.templates.PynqBackend.make_bitfile(hls_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default = "baseline.yml", help="specify yaml config")

    args = parser.parse_args()

    main(args)
