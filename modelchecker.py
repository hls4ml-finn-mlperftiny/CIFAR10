from itertools import product
import csv
from resnet_v1_eembc import resnet_v1_eembc

filters = [2]  # , 4, 8, 16]
kernelsizes = [1, 2, 3]
strides1 = ['111', '211', '244', '311', '334', '343', '344', '411', '424', '433', '434', '442', '443', '444']
strides23 = ['111', '122', '133', '144', '212', '224', '313', '414']

configs_comb = product(filters, filters,  # block 1
                       filters, filters,  # block 2
                       filters, filters,  # block 3
                       kernelsizes, kernelsizes, kernelsizes,  # block 1
                       kernelsizes, kernelsizes, kernelsizes,  # block 2
                       kernelsizes, kernelsizes, kernelsizes,  # block 3
                       strides1,  # block 1
                       strides23,  # block 2
                       strides23  # block 3
                       )
configs_comb = list(configs_comb)
valid_configs = []
for c in configs_comb:

    try:
        model = resnet_v1_eembc(num_filters=c[0:6], kernel_sizes=c[6:15], strides=c[15:])
    except ValueError:
        print(c, "not valid")
        continue
    print(c, "valid")
    valid_configs.append(c)

print("valid configs")
print(valid_configs)
