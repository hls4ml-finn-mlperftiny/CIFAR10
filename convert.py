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

    _, (X_test, y_test) = cifar10.load_data()
    X_test = np.ascontiguousarray(X_test[:100]/256.)
    num_classes = 10
    y_test = tf.keras.utils.to_categorical(y_test[:100], num_classes)

    y_keras = model.predict(X_test)

    np.save('y_keras.npy', y_keras)
    np.save('y_test.npy', y_test)
    np.save('X_test.npy', X_test)

    import hls4ml
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model'] = {}
    config['Model']['ReuseFactor'] = our_config['convert']['ReuseFactor']
    config['Model']['Strategy'] = our_config['convert']['Strategy']
    config['Model']['Precision'] = our_config['convert']['Precision']
    for name in config['LayerName'].keys():
        config['LayerName'][name]['Trace'] = bool(our_config['convert']['Trace'])
        config['LayerName'][name]['ReuseFactor'] = our_config['convert']['ReuseFactor']
        config['LayerName'][name]['Precision'] = our_config['convert']['Precision']
        if 'activation' in name:
            config['LayerName'][name]['Precision'] = our_config['convert']['PrecisionActivation']
    # custom configs
    for name in our_config['convert']['Override'].keys():
        config['LayerName'][name].update(our_config['convert']['Override'][name])

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

    # profiling / testing
    hls_model = hls4ml.converters.keras_to_hls(cfg)

    hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='model_hls4ml.png')

    if bool(our_config['convert']['Trace']):
        from hls4ml.model.profiling import compare, numerical
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.figure()
        wp, ap = numerical(model=model, hls_model=hls_model, X=X_test)
        plt.show()
        plt.savefig('profiling_numerical.png', dpi=300)
        
        plt.figure()
        cp = compare(keras_model=model, hls_model=hls_model, X=X_test, plot_type="dist_diff")
        plt.show()
        plt.savefig('profiling_compare.png', dpi=300)

        y_hls, hls4ml_trace = hls_model.trace(X_test)
        np.save('y_hls.npy', y_hls)
        keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, X_test)
        
        for layer in hls4ml_trace.keys():
            plt.figure()
            plt.scatter(hls4ml_trace[layer].flatten(), keras_trace[layer].flatten(), s=0.2)
            min_x = min(np.amin(hls4ml_trace[layer]), np.amin(keras_trace[layer]))
            max_x = max(np.amax(hls4ml_trace[layer]), np.amax(keras_trace[layer]))
            plt.plot([min_x, max_x], [min_x, max_x], c='gray')
            plt.xlabel('hls4ml {}'.format(layer))
            plt.ylabel('QKeras {}'.format(layer))
            plt.show()
            plt.savefig('profiling_{}.png'.format(layer), dpi=300)
    else:
        hls_model.compile()
        y_hls = hls_model.predict(X_test)

    from sklearn.metrics import accuracy_score
    print("Keras Accuracy:  {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))
    print("hls4ml Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))))

    # Bitfile time 
    #hls_model.build(csim=False,synth=True,export=True)
    #hls4ml.report.read_vivado_report(our_config['convert']['OutputDir'])
    #if our_config['convert']['Backend'] == 'Pynq':
    #    hls4ml.templates.PynqBackend.make_bitfile(hls_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default = "baseline.yml", help="specify yaml config")

    args = parser.parse_args()

    main(args)
