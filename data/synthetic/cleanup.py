import math

'''
A temporary script to find out which outliers are actually lying
inside the blobs
'''

FILE = 'set_950_50_2_l.csv'

C1 = (-2, -2)
C2 = (2, 2)
R1 = 0.8
R2 = 1.5


def dist(x, center):
    return math.sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2)


with open(FILE, 'r') as f:
    for line in f:
        split = line.split(',')
        x = (float(split[0]), float(split[1]))
        if split[-1] == '1.000000000000000000e+00\n' and (dist(x, C1) < R1 or dist(x, C2) < R2):
            print(line, end='')
