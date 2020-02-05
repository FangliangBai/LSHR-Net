from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from pandas import DataFrame
import csv

ea = event_accumulator.EventAccumulator(
    '/media/kent/DISK2/sr_spc/experiment_53/models/events.out.tfevents.1526384994.kent-System-Product-Name',
    size_guidance={  # see below regarding this argument
        event_accumulator.COMPRESSED_HISTOGRAMS: 1,
        event_accumulator.IMAGES: 1,
        event_accumulator.AUDIO: 1,
        event_accumulator.SCALARS: 1,
        event_accumulator.HISTOGRAMS: 0,
    })

ea.Reload()  # loads events from file

b = ea.Histograms('sr_spc/linear_mapping/binary_kernels')

binary_array = np.ndarray(shape=[len(b), 2], dtype=np.int)

for i in range(len(b)):
    histogram = b[i]
    binary_array[i, 0] = histogram.histogram_value.bucket[1]
    binary_array[i, 1] = histogram.histogram_value.bucket[3]

binary_array_1 = binary_array[0:len(b)-1, :]
binary_array_2 = binary_array[1:, :]

change = binary_array_2 - binary_array_1
change = np.abs(change)

csv_data = DataFrame(change, columns=['1', '0'])
csv_data.index = csv_data.index + 1
csv_data.to_csv('binary_change.csv')


# number of non-zero weights
weight_one = binary_array[:, 1]
weight_one = np.ceil(weight_one)

csv_weight_one = DataFrame(weight_one, columns=['p'])
csv_weight_one.index = csv_weight_one.index + 1
csv_weight_one.to_csv('binary_weight_one.csv')
