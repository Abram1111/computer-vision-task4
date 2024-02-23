# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import sys
# import random
# import numpy as np
# from cmath import exp, pi


# def get_window_width_from_input(values_count):
#     window_width_power = int(raw_input('Enter the window width (power of 2, between 1 and 16): '))

#     if window_width_power not in range(1, 16):
#         print 'Window width power of 2 must be integer between 1 and 16, "{}" given.'.format(window_width_power)
#         print 'Exiting.'

#         sys.exit(1)

#     window_width = pow(2, window_width_power)

#     print 'Window width: {}'.format(window_width)

#     if values_count < window_width:
#         print 'Window width must not be larger than values count!'
#         print 'Exiting.'

#         sys.exit(1)

#     return window_width


# def read_data(filename):
#     try:
#         f = open(filename, 'r')

#     except (IOError, OSError):
#         print 'Can\'t open file "{}"'.format(filename)
#         sys.exit(1)

#     values = f.read().split()

#     f.close()

#     for i, value in enumerate(values):
#         values[i] = float(value)

#     return values


# def write_data(filename, data):
#     try:
#         f = open(filename, 'w')

#     except (IOError, OSError):
#         print 'Can\'t open file "{}"'.format(filename)
#         sys.exit(1)

#     f.write(data)
#     f.close()


# def get_windows(values, width):
#     count = 2 * (len(values) - width) / width + 1

#     print 'Number of windows: {}'.format(count)

#     results = {}

#     for i in xrange(0, count):
#         results[i] = {}

#         for j in xrange(0, width):
#             results[i][j] = values[j + i * width / 2]

#     return results


# def fft(values):
#     values_count = len(values)

#     if np.log2(values_count) % 1 > 0:
#         raise ValueError('values count must be a power of 2, "{}" given.'.format(values_count))

#     t = exp(-2 * pi * 1j / values_count)

#     if values_count > 1:
#         values = fft(values[::2]) + fft(values[1::2])

#         for k in range(values_count // 2):
#             k_value = values[k]
#             values[k] = k_value + t ** k * values[k + values_count // 2]
#             values[k + values_count // 2] = k_value - t ** k * values[k + values_count // 2]

#     return values


# # def __main__():
#     data = read_data(raw_input('Enter input file path: '))
#     # data = np.random.random(pow(2, 4))
#     # data = []
#     # for data_key in range(0, pow(2, 2)):
#     #     data.append(round(random.uniform(-1.0, 1.0), 8))

#     print 'Values count: {}'.format(len(data))

#     window_width = get_window_width_from_input(len(data))

#     windows = get_windows(data, window_width)

#     output_data = ''

#     for window_key, window in windows.iteritems():
#         # for value in np.fft.fft(window.values())
#         output_values = []

#         fft_values = fft(window.values())

#         half_of_fft_values = fft_values[:len(fft_values) / 2]

#         for value in half_of_fft_values:
#             output_values.append(str(np.abs(value)))

#         output_data += ', '.join(output_values)
#         output_data += '\n'

#     write_data(raw_input('Enter output file path: '), output_data)

#     print 'FFT done!'