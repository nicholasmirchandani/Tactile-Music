#REQUIREMENTS:
#pip install scipy
#pip install matplotlib

from scipy.io import wavfile
from scipy.fft import fft
from scipy import signal
import math
import numpy
import matplotlib.pyplot as pyplot

#Read in data
samplerate, data = wavfile.read('test.wav')
#If 2 channel audio, take a channel and process it as mono
if(len(data.shape) >= 2 and data.shape[1] == 2):
    temp_data = []
    for i in range(0, len(data)):
        temp_data.append(data[i][0])
    data = temp_data

#Apply filter, and fix output type
lp1 = [0.127174276079605, 0.0581343489943583, 0.0681122463081755, 0.0766052817881472, 0.0830675938972334, 0.0871853443909994, 0.0884935091352945, 0.0871853443909994, 0.0830675938972334, 0.0766052817881472, 0.0681122463081755, 0.0581343489943583, 0.127174276079605]
bp1 = [0.0109723768383746, -0.0467943943338264, -0.0741398108994016, -0.149777301781025, 0.117993634189359, 0.192388845547486, 0.294512671843853, 0.192388845547486, 0.117993634189359, -0.149777301781025, -0.0741398108994016, -0.0467943943338264, 0.0109723768383746]
hp1 = [-0.0351103427314022, 0.120418583869658, 0.0883153039547716, 0.00865009773730016, -0.134411547496756, -0.277541793649009, 0.662413172546772, -0.277541793649009, -0.134411547496756, 0.00865009773730016, 0.0883153039547716, 0.120418583869658, -0.0351103427314022]
id1 = [0.5]
newdata = signal.convolve(data, id1)
#Fix newdata datatype to original datatype
newdata = newdata.astype(type(data[0]))

#Write to output file to ensure nothing broke.  Calculate FFTs
wavfile.write('output.wav', samplerate, newdata)

#Converting data to all be within -1 to 1, so fft calculations are consistent across filetypes.
if(type(data[0]) == numpy.float32):
    #At the moment float32 is the desired range of -1 to 1
    pass
elif(type(data[0]) == numpy.int16):
    #int16 is -32768 to 32767
    data = numpy.array(data)
    data = data.astype(numpy.float32)
    newdata = newdata.astype(numpy.float32)
    data = data / 32768
    newdata = newdata / 32768
elif(type(data[0]) == numpy.int32):
    #int32 is -2147483648 to 2147483647
    data = numpy.array(data)
    data = data.astype(numpy.float32)
    newdata = newdata.astype(numpy.float32)
    data = data / 2147483648
    newdata = newdata / 2147483648
elif(type(data[0]) == numpy.uint8):
    #uint is 0 to 255
    data = numpy.array(data)
    data = data.astype(numpy.float32)
    newdata = newdata.astype(numpy.float32)
    data = (data / 128) - 1
    newdata = (newdata / 128) - 1

data_fft = fft(data, samplerate)
newdata_fft = fft(newdata, samplerate)

#Multiply by the conjugate element by element to get the power, removing all imaginary components.
for i in range(0, len(data_fft)):
    data_fft[i] = data_fft[i] * numpy.conj(data_fft[i])

for i in range(0, len(newdata_fft)):
    newdata_fft[i] = newdata_fft[i] * numpy.conj(newdata_fft[i])

minVal = 9999
maxVal = -9999
for i in range(0, len(newdata)):
    if(newdata[i] > maxVal):
        maxVal = newdata[i]
    if(newdata[i] < minVal):
        minVal = newdata[i]

print("NEW MIN: ", minVal, "\nNEW MAX: ", maxVal)

#Dividing by the length of the field so we can be normalized across datasets
data_fft = data_fft / (len(data))
newdata_fft = newdata_fft / (len(newdata))

#Taking the square root of ffts so squaring of power to remove imaginary components doesn't break further math
data_fft = data_fft ** (1/2)
newdata_fft = newdata_fft ** (1/2)

#Using math.log10 because np.log10 was giving errors
data_decibel = []
newdata_decibel = []
for i in range(0, len(data_fft)):
    data_fft[i] = numpy.real(data_fft[i])
    data_decibel.append(10 * math.log10(data_fft[i]))
for i in range(0, len(newdata_fft)):
    newdata_fft[i] = numpy.real(newdata_fft[i])
    newdata_decibel.append(10 * math.log10(newdata_fft[i]))

print(type(data_fft[0]))
print("DATA FFT: ", data_fft)


#Decibel maths
#math.log10
#data_decibel = 10*numpy.log10(data_fft)
#newdata_decibel = 10*numpy.log10(newdata_fft)

#Plot Results!
fig, axs = pyplot.subplots(2, 2)
axs[0,0].plot(data)
axs[0,0].set_title('Old numbers')

axs[0,1].plot(newdata)
axs[0,1].set_title('New numbers')

#setting min and max so we have some bounds to relatively compare everything
ymin = -50
ymax = 50

#Using samplerate/2 because Nyquist rate only ensures everything up to that point is accurate
axs[1,0].plot(data_decibel[0:(int)(samplerate/2)])
axs[1,0].set_title('Input Loudness')
axs[1,0].set_ylim(pyplot.ylim(bottom=ymin, top=ymax))

axs[1,1].plot(newdata_decibel[0:(int)(samplerate/2)])
axs[1,1].set_title('Output Loudness')
axs[1,1].set_ylim(pyplot.ylim(bottom=ymin, top=ymax))

pyplot.figure(2)
data_decibel = numpy.array(data_decibel)
newdata_decibel = numpy.array(newdata_decibel)
#pyplot.plot(newdata_decibel[0:(int)(samplerate/2)]-data_decibel[0:(int)(samplerate/2)])
pyplot.plot(newdata_fft[0:(int)(samplerate/2)]/data_fft[0:(int)(samplerate/2)])
pyplot.gca().set_ylim([0,10])

pyplot.show()
