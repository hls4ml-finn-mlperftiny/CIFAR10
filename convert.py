import os
# edit depending on where Vivado is installed:
# os.environ['PATH'] = '/<Xilinx installation directory>/Vivado/<version>/bin:' + os.environ['PATH']
os.environ['PATH'] = '/opt/local/Xilinx/Vivado/2019.2/bin:' + os.environ['PATH']
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
from tensorflow.keras.datasets import cifar10
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
    our_config = yaml_load(args.config)
    save_dir = our_config['save_dir']
    model_name = our_config['model']['name']
    model_file_path = os.path.join(save_dir, 'model_best.h5')

    from tensorflow.keras.models import load_model
    from qkeras.utils import _add_supported_quantized_objects
    co = {}
    _add_supported_quantized_objects(co)    

    model = load_model(model_file_path, custom_objects=co)
    model.summary()

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_test = np.ascontiguousarray(X_test[:100]/255.)
    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test[:100], num_classes)

    y_keras = model.predict(X_test)

    np.save('y_keras.npy', y_keras)
    np.save('y_test.npy', y_test)
    np.save('X_test.npy', X_test)

    import hls4ml
    hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
    hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
    hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model'] = {}
    config['Model']['ReuseFactor'] = our_config['convert']['ReuseFactor']
    config['Model']['Strategy'] = our_config['convert']['Strategy']
    config['Model']['Precision'] = our_config['convert']['Precision']
    for name in config['LayerName'].keys():
        config['LayerName'][name]['Trace'] = True
        config['LayerName'][name]['ReuseFactor'] = our_config['convert']['ReuseFactor']
        config['LayerName'][name]['Precision'] = our_config['convert']['Precision']
        if 'normalization' in name:
            config['LayerName'][name]['Precision'] = 'ap_fixed<16,4>'
        elif 'activation' in name:
            config['LayerName'][name]['Precision'] = 'ap_fixed<10,4>'
    # custom config for softmax
    config['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<18,8>'
    config['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,4>'
    config['LayerName']['softmax']['Strategy'] = 'Stable'
    config['LayerName']['softmax']['Precision'] = 'ap_fixed<18,2>'


    cfg = hls4ml.converters.create_backend_config(fpga_part='xc7z020clg400-1')
    cfg['HLSConfig'] = config
    cfg['IOType'] = our_config['convert']['IOType']
    cfg['Backend'] = our_config['convert']['Backend']
    cfg['Interface'] = 's_axilite' # or 'm_axi'
    cfg['ClockPeriod'] = our_config['convert']['ClockPeriod']
    cfg['KerasModel'] = model
    cfg['OutputDir'] = our_config['convert']['OutputDir']

    print("-----------------------------------")
    print_dict(cfg)
    print("-----------------------------------")

    # Bitfile time 
    hls_model = hls4ml.converters.keras_to_hls(cfg)
    hls_model.compile()
    y_hls = hls_model.predict(X_test.astype(dtype=np.float32))
    np.save('y_hls.npy', y_hls)

    from hls4ml.model.profiling import compare, numerical
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wp, ap = numerical(model=model, hls_model=hls_model, X=X_test)
    plt.show()
    plt.savefig('profiling_numerical.png', dpi=300)

    cp = compare(keras_model=model, hls_model=hls_model, X=X_test, plot_type="dist_diff")
    plt.show()
    plt.savefig('profiling_compare.png', dpi=300)

    hls4ml_pred, hls4ml_trace = hls_model.trace(X_test)
    keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, X_test)

    from sklearn.metrics import accuracy_score
    print("Keras Accuracy:  {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))
    print("hls4ml Accuracy: {}".format(accuracy_score(np.argmax(y_hls, axis=1), np.argmax(y_keras, axis=1))))

    #hls_model.build(csim=False,synth=True,export=True)
    #hls4ml.report.read_vivado_report(our_config['convert']['OutputDir'])
    #if our_config['convert']['Backend'] == 'Pynq':
    #    hls4ml.templates.PynqBackend.make_bitfile(hls_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default = "baseline.yml", help="specify yaml config")

    args = parser.parse_args()

    main(args)
