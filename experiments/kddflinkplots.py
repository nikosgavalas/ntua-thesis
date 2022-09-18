#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

# |G|25|8|12|18|
# |G|50|8|12|20|
# |G|75|8|1|20|
# |G|100|8|12,1,1,12|26,21,21,23|
# |IF|25|8|11|22|
# |IF|50|8|1|31|
# |IF|75|8|12|42|
# |IF|100|8|1|51|
# |HS|25|8|12|31|
# |HS|50|8|1|39|
# |HS|75|8|11|58|
# |HS|100|8|1|70|
# |LD|25|8|1|22|
# |LD|50|8|1|27|
# |LD|75|8|12|44|
# |LD|100|8|1|45|

# |G|100|1|1|72|
# |G|100|2|1|36|
# |G|100|3|6|30|
# |G|100|4|1|29|
# |G|100|8|12,1,1,12|26,21,21,23|

# |IF|100|1|1|110|***
# |IF|100|2|12|96|
# |IF|100|3|12|67|
# |IF|100|4|1|61|
# |IF|100|8|1|51|

# |HS|100|1|6|156|
# |HS|100|2|11|103,75,88,99|
# |HS|100|3|1|60,160|
# |HS|100|4|1|53,93|
# |HS|100|8|12|69|

# |LD|100|1|6|119|
# |LD|100|2|1|71|
# |LD|100|3|1|53|
# |LD|100|4|12|78|
# |LD|100|8|1|45|

# par = 8
# read + stream time = ~at most 12sec, pipelined so we dont care
values_sizes = [1224607, 2449215, 3673823, 4898431]
# values = [25, 50, 75, 100]
# times_gauss = [18, 20, 21, 23]
# times_iforest = [22, 31, 42, 51]
# times_hstrees = [31, 39, 58, 70]
# times_loda = [22, 27, 44, 45]

# size=full
# read + stream time = ~at most 12sec, pipelined so we dont care
values_sizes = [4898431, 4898431, 4898431, 4898431]
values = [1, 2, 4, 8]
times_gauss = [72, 36, 29, 23]
times_iforest = [110, 96, 61, 51]
times_hstrees = [156, 103, 93, 70]
times_loda = [119, 71, 78, 45]

sizes = np.array(values_sizes)
throughput_gauss = sizes / np.array(times_gauss)
throughput_iforest = sizes / np.array(times_iforest)
throughput_hstrees = sizes / np.array(times_hstrees)
throughput_loda = sizes / np.array(times_loda)

plt.plot(values, throughput_gauss, c='green', marker='o',
         label='gauss', linestyle=(0, ()))
plt.plot(values, throughput_iforest, c='red', marker='^',
         label='iforest')
plt.plot(values, throughput_hstrees, c='blue', marker='s',
         label='hstrees', linestyle=(0, (1, 1)))
plt.plot(values, throughput_loda, c='black', marker='D',
         label='loda', linestyle=(0, (5, 5)))

plt.ylabel('Throughput (examples per second)')
plt.xlabel('Parallelism')
#plt.xticks(rotation=20)

plt.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2),
           mode="expand", borderaxespad=0, ncol=4)
plt.show()
