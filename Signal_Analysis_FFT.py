'''
# author : Thomas Georgiadis
#
# File Signal_Analysis_FFT.py
#
# File Description : Takes a csv file containing vibration measurements and plots
#                    the time and frequency responses.
#
'''
from scipy.fft import fft
from pylab import *
import numpy as np
import pylab as pl
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Importing measurements.

data = pd.read_csv("temp.csv")
x_data = data.drop(columns=['y'])
y_data = data.drop(columns=['x'])
# z_data = data.drop(columns=['x','y'])

signal_x = x_data.to_numpy().reshape([x_data.size,])
signal_y = y_data.to_numpy().reshape([y_data.size,])
# signal_z = z_data.to_numpy().reshape([z_data.size,])

# sample spacing

T = 1.0 / x_data.size

# Peak frequencies detection variables.

freq_offset = 2 #removing very low frequencies where all the power is gathered
max_freqs_x = np.empty([5,])

##############################################################################

# Fourier analysis of x axis vibrations.

# Number of sample points

N = x_data.size

x = np.linspace(0.0, N*T, N)

#y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
#print(y.shape)
y = signal_x
#print(signal_x.shape)
yf = fft(y)

xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

plt_fft_x = 2.0/N * np.abs(yf[0:N//2])

plt.figure(1)

plt.subplot(2, 1, 1)
plt.plot(signal_x)
plt.ylabel('Time Response (ms)')

plt.grid()

plt.subplot(2, 1, 2)
plt.plot(xf, plt_fft_x)
plt.ylabel('Frequency Response (Hz)')
plt.grid()
xlim(0,N/2)
plt.show()

index_old = 1000

peak_search_arr = list(plt_fft_x[freq_offset:])
for i in range(0,4):
    max_num = max(peak_search_arr)

    index = peak_search_arr.index(max_num)
    if index_old <= index :
        index = index + i
    index_old = index

    print(xf[index+freq_offset])
    max_freqs_x[i] = xf[index+freq_offset]
    peak_search_arr.remove(max_num)
print(max_freqs_x)

# Fourier analysis of y axis vibrations.

# Number of sample points

Ny = y_data.size

x = np.linspace(0.0, Ny*T, Ny)

y = signal_y
yf = fft(y)

xf = np.linspace(0.0, 1.0/(2.0*T), Ny//2)

plt_fft_y = 2.0/N * np.abs(yf[0:N//2]);

plt.figure(2)

plt.subplot(2, 1, 1)
plt.plot(signal_y)
# plt.xlabel('Time(ms)')
# plt.ylabel('Amplitude (g)')
plt.ylabel('Time Response (ms)')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(xf, plt_fft_y)
plt.grid()
xlim(0,N/2)
# plt.xlabel('Frequency(Hz)')
# plt.ylabel('Amplitude')
plt.ylabel('Frequency Response (Hz)')
plt.show()

index_old = 1000

peak_search_arr = list(plt_fft_y[freq_offset:])
for i in range(0,4):
    max_num = max(peak_search_arr)

    index = peak_search_arr.index(max_num)
    if index_old <= index :
        index = index + i
    index_old = index

    print(xf[index+freq_offset])
    max_freqs_x[i] = xf[index+freq_offset]
    peak_search_arr.remove(max_num)
print(max_freqs_x)

# Fourier analysis of z axis vibrations.

# Number of sample points
'''
Nz = z_data.size

x = np.linspace(0.0, Nz*T, Nz)

y = signal_z
yf = fft(y)

xf = np.linspace(0.0, 1.0/(2.0*T), Nz//2)
plt_fft_z = 2.0/N * np.abs(yf[0:N//2]);

plt.figure(3)

plt.subplot(2, 1, 1)
plt.plot(signal_z)
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(xf, plt_fft_z)
plt.grid()
xlim(0,200)
plt.show()

index_old = 1000

peak_search_arr = list(plt_fft_z[freq_offset:])

for i in range(0,4):
    max_num = max(peak_search_arr)

    index = peak_search_arr.index(max_num)
    if index_old <= index :
        index = index + i
    index_old = index

    print(xf[index+freq_offset])
    max_freqs_x[i] = xf[index+freq_offset]
    peak_search_arr.remove(max_num)
print(max_freqs_x)

'''
