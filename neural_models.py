import scipy.io
import numpy as np
import string
import matlab.engine
import data_holder as data
try:
    eng = matlab.engine.connect_matlab(matlab.engine.find_matlab()[0])
except IndexError:
    eng = matlab.engine.start_matlab()

savepath = 'C:\\Users\\Flynn\\OneDrive\\Documents\\MATLAB\\neuroexplore\\'
letters = list(string.ascii_uppercase)

data = data.Data().data
trials = { k: v for k, v in data['trials'].items() if len(v) > 0 }
spike_times_dict = { # ugly but works : filters dataframe to an array of the first column
    k1: {str(k2): v2['neuron'].values for k2, v2 in v1.items()} for k1, v1 in trials.items()
}

e_BR_dict = spike_times_dict['e_BR']
e_BL_dict = spike_times_dict['e_BL']
e_FL_dict = spike_times_dict['e_FL']

for i, key in enumerate(list(e_BR_dict.keys())):
    e_BR_dict[letters[i]] = e_BR_dict.pop(key)

mat_struct = scipy.io.matlab.mio5_params.mat_struct()
for key, value in e_BR_dict.items():
    mat_struct.__setattr__(key, value[:170])

scipy.io.savemat(savepath + 'e_BR.mat', {'e_BR': mat_struct})


# final = {}
#
# for k, v in spike_times_dict.items(): # k = e_BR, v = {3: array, 4: array}
#     for outer, inner in v.items(): # outer = 3, inner = array
#         for k2, v2 in inner.items():
#
#         mat_structs.__setattr__(outer, inner)
#     final[k] = mat_structs
#

# scipy.io.savemat(savepath + 'from_python.mat', final)

