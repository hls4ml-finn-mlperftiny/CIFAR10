from itertools import product
import csv
from resnet_v1_eembc import resnet_v1_eembc

filters0 = [2, 4, 8, 16, 32, 64]
filters1 = [0] #2, 4, 8, 16, 32, 64]
filters2 = [0] #2, 4, 8, 16, 32, 64]
kernelsize0 = [1, 2, 3]
kernelsize1 = [0] #[1, 2, 3]
strides0 = [1, 4]
strides1 = [0] #[1, 2, 3, 4]

configs_comb = product(filters0, filters1, filters2, kernelsize0, kernelsize1, strides0, strides1)
configs_comb = list(configs_comb)
valid_configs = []
for c in configs_comb:

    try:
        model = resnet_v1_eembc(num_filters=c[0:3], kernel_sizes=c[3:5], strides=c[5:7])
    except ValueError:
        print(c, "not valid")
        continue
    print(c, "valid")
    valid_configs.append(c)

print("valid configs")
print(valid_configs)
