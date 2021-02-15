from itertools import product
import csv

filters_InL = [4,8,16,32,64]
kernelsize_InL = [1,2,3,4]
strides_InL = [1,2,3,4]

filters_L1 = [4,8,16,32,64]
kernelsize_L1 = [1,2,3,4]
strides_L1 = [1,2,3,4]

filters_L2 = [4,8,16,32,64]
kernelsize_L2 = [1,2,3,4]
strides_L2 = [1,2,3,4]

filters_L3 = [4,8,16,32,64]
kernelsize_L3 = [1,2,3,4]
strides_L3 = [1,2,3,4]

configs_comb = product(filters_InL,kernelsize_InL,strides_InL)  #,filters_L1,kernelsize_L1,strides_L1,filters_L2,kernelsize_L2,strides_L2)
configs_comb = list(configs_comb)

print(configs_comb)
print("\n",len(configs_comb))

#initialise .csv requirements
csvfields = ['Filters_IL','Kernel_IL','Accuracy_IL']
csvfilename = "Valid_hyper/IL_only.csv"            #CHANGE THESE DEPENDING ON WHAT YOU'RE SAVING AND WHERE YOU'RE SAVING
#csvdata_rows = [['Layer 2 kernels at 2 from 3 ad Input Layer Stride 2 patience 30',total_flop/1e9,evaluation[1]]]
with open(csvfilename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(csvfields)
    csvwriter.writerows(csvdata_rows)