#REQUIREMENTS:
#pip install scipy
#pip install matplotlib
#pip install simpleaudio

from scipy.io import wavfile
from scipy.fft import fft
from scipy import signal
import math
import numpy
import matplotlib.pyplot as pyplot
import simpleaudio
import time

#Read in data
samplerate, data = wavfile.read('test.wav')
#If 2 channel audio, take a channel and process it as mono
if(len(data.shape) >= 2 and data.shape[1] == 2):
    temp_data = []
    for i in range(0, len(data)):
        temp_data.append(data[i][0])
    data = numpy.array(temp_data)

rawdata = data #Saving rawdata for playback

#Refactoring Constants
LOW_PASS = 0
BAND_PASS = LOW_PASS + 1
HIGH_PASS = BAND_PASS + 1
GAIN_REDUCTION = HIGH_PASS + 1
NUM_FILTERS = GAIN_REDUCTION + 1

#Refactoring Dictionaries
filters = {}
filtered_data = {}
fft_data = {}
intensities = {}

#Apply filters, and fix output type
filters[LOW_PASS] = [0.127174276079605, 0.0581343489943583, 0.0681122463081755, 0.0766052817881472, 0.0830675938972334, 0.0871853443909994, 0.0884935091352945, 0.0871853443909994, 0.0830675938972334, 0.0766052817881472, 0.0681122463081755, 0.0581343489943583, 0.127174276079605]
filters[BAND_PASS] = [0.0109723768383746, -0.0467943943338264, -0.0741398108994016, -0.149777301781025, 0.117993634189359, 0.192388845547486, 0.294512671843853, 0.192388845547486, 0.117993634189359, -0.149777301781025, -0.0741398108994016, -0.0467943943338264, 0.0109723768383746]
filters[HIGH_PASS] = [-0.0351103427314022, 0.120418583869658, 0.0883153039547716, 0.00865009773730016, -0.134411547496756, -0.277541793649009, 0.662413172546772, -0.277541793649009, -0.134411547496756, 0.00865009773730016, 0.0883153039547716, 0.120418583869658, -0.0351103427314022]
filters[GAIN_REDUCTION] = [0.5]

filtered_data[LOW_PASS] = signal.convolve(data, filters[LOW_PASS])
filtered_data[BAND_PASS] = signal.convolve(data, filters[BAND_PASS])
filtered_data[HIGH_PASS] = signal.convolve(data, filters[HIGH_PASS])
filtered_data[GAIN_REDUCTION] = signal.convolve(data, filters[GAIN_REDUCTION])

#Fix filtered datatypes to original datatype
filtered_data[LOW_PASS] = filtered_data[LOW_PASS].astype(type(data[0]))
filtered_data[BAND_PASS] = filtered_data[BAND_PASS].astype(type(data[0]))
filtered_data[HIGH_PASS] = filtered_data[HIGH_PASS].astype(type(data[0]))
filtered_data[GAIN_REDUCTION] = filtered_data[GAIN_REDUCTION].astype(type(data[0]))

#No longer need to write to output file to ensure nothing broke.
# wavfile.write('output.wav', samplerate, filtered_data[GAIN_REDUCTION])

bytesPerSample = 2

#Converting data to all be 32 bit floats within -1 to 1, so fft calculations are consistent across filetypes.
filtered_data[LOW_PASS] = filtered_data[LOW_PASS].astype(numpy.float32)
filtered_data[BAND_PASS] = filtered_data[BAND_PASS].astype(numpy.float32)
filtered_data[HIGH_PASS] = filtered_data[HIGH_PASS].astype(numpy.float32)
filtered_data[GAIN_REDUCTION] = filtered_data[GAIN_REDUCTION].astype(numpy.float32)

if(type(data[0]) == numpy.float32):
    #Float32 is 4 bytes in the desired range of -1 to 1
    #rawdata *= 2147483647 #Transforming rawdata to integer array
    #rawdata = rawdata.astype(int)
    bytesPerSample = 4
elif(type(data[0]) == numpy.int16):
    #int16 is 2 bytes -32768 to 32767
    data = numpy.array(data)
    data = data.astype(numpy.float32)
    data = data / 32768
    filtered_data[LOW_PASS] = filtered_data[LOW_PASS] / 32768
    filtered_data[BAND_PASS] = filtered_data[BAND_PASS] / 32768
    filtered_data[HIGH_PASS] = filtered_data[HIGH_PASS] / 32728
    filtered_data[GAIN_REDUCTION] = filtered_data[GAIN_REDUCTION] / 32768
elif(type(data[0]) == numpy.int32):
    #int32 is 4 bytes -2147483648 to 2147483647
    data = numpy.array(data)
    data = data.astype(numpy.float32)
    data = data / 2147483648
    filtered_data[LOW_PASS] = filtered_data[LOW_PASS] / 2147483648
    filtered_data[BAND_PASS] = filtered_data[BAND_PASS] / 2147483648
    filtered_data[HIGH_PASS] = filtered_data[HIGH_PASS] / 2147483648
    filtered_data[GAIN_REDUCTION] = filtered_data[GAIN_REDUCTION] / 2147483648
    bytesPerSample = 4
elif(type(data[0]) == numpy.uint8):
    #uint is 1 byte 0 to 255
    data = numpy.array(data)
    data = data.astype(numpy.float32)
    data = (data / 128) - 1
    filtered_data[LOW_PASS] = (filtered_data[LOW_PASS] / 128) - 1
    filtered_data[BAND_PASS] = (filtered_data[BAND_PASS] / 128) - 1
    filtered_data[HIGH_PASS] = (filtered_data[HIGH_PASS] / 128) - 1
    filtered_data[GAIN_REDUCTION] = (filtered_data[GAIN_REDUCTION] / 128) - 1
    bytesPerSample = 1

interval = .2 #Interval is .2 sec to start to leave computation time
samplesPerInterval = math.ceil(samplerate * interval) #NOTE: This rounds up, so in instances where samplerate * interval isn't an integer, there may be desync issues, although with conventionally large sampling rates and a clean interval like .2 (aka divide by 5) that shouldn't be a problem.
numSegments = math.ceil(len(data) / samplesPerInterval)

#TODO: Delay play by one interval because that's the delay of the signal processing
play_obj = simpleaudio.play_buffer(rawdata, 1, bytesPerSample, samplerate)

#NOTE: Since samplesPerInterval is dependent on samplerate, with higher sample rate audio, there is a significant performance hit.  Audio should be downsampled, either in code (NYI) or in Audacity, before processing
for segment in range(0,numSegments):
    starttime = time.time()

    #Calculate the ffts of specifically the desired slice of time using some simple indexing
    data_fft = fft(data[(segment * samplesPerInterval):((segment+1) * samplesPerInterval)], samplerate)
    fft_data[LOW_PASS] = fft(filtered_data[LOW_PASS][(segment * samplesPerInterval):((segment+1) * samplesPerInterval)], samplerate)
    fft_data[BAND_PASS] = fft(filtered_data[BAND_PASS][(segment * samplesPerInterval):((segment+1) * samplesPerInterval)], samplerate)
    fft_data[HIGH_PASS] = fft(filtered_data[HIGH_PASS][(segment * samplesPerInterval):((segment+1) * samplesPerInterval)], samplerate)
    fft_data[GAIN_REDUCTION] = fft(filtered_data[GAIN_REDUCTION][(segment * samplesPerInterval):((segment+1) * samplesPerInterval)], samplerate)

    #Multiply by the conjugate element by element to get the power, removing all imaginary components.
    #NOTE: Loops are all separate to allow filters to be completely independent, with an arbitrary amount of filter coefficients
    for i in range(0, len(data_fft)):
        data_fft[i] = data_fft[i] * numpy.conj(data_fft[i])

    for i in range(0, len(fft_data[LOW_PASS])):
        fft_data[LOW_PASS][i] = fft_data[LOW_PASS][i] * numpy.conj(fft_data[LOW_PASS][i])

    for i in range(0, len(fft_data[BAND_PASS])):
        fft_data[BAND_PASS][i] = fft_data[BAND_PASS][i] * numpy.conj(fft_data[BAND_PASS][i])

    for i in range(0, len(fft_data[HIGH_PASS])):
        fft_data[HIGH_PASS][i] = fft_data[HIGH_PASS][i] * numpy.conj(fft_data[HIGH_PASS][i])

    for i in range(0, len(fft_data[GAIN_REDUCTION])):
        fft_data[GAIN_REDUCTION][i] = fft_data[GAIN_REDUCTION][i] * numpy.conj(fft_data[GAIN_REDUCTION][i])

    #Dividing by the length of the field so we can be normalized across datasets
    data_fft = data_fft / (len(data))
    fft_data[LOW_PASS] = fft_data[LOW_PASS] / (len(filtered_data[LOW_PASS]))
    fft_data[BAND_PASS] = fft_data[BAND_PASS] / (len(filtered_data[BAND_PASS]))
    fft_data[HIGH_PASS] = fft_data[HIGH_PASS] / (len(filtered_data[HIGH_PASS]))
    fft_data[GAIN_REDUCTION] = fft_data[GAIN_REDUCTION] / (len(filtered_data[GAIN_REDUCTION]))

    #Taking the square root of ffts so squaring of power to remove imaginary components doesn't break further math
    #Necessary so a 2x gain reduction is a 2x intensity reduction
    data_fft = data_fft ** (1/2)
    fft_data[LOW_PASS] = fft_data[LOW_PASS] ** (1/2)
    fft_data[BAND_PASS] = fft_data[BAND_PASS] ** (1/2)
    fft_data[HIGH_PASS] = fft_data[HIGH_PASS] ** (1/2)
    fft_data[GAIN_REDUCTION] = fft_data[GAIN_REDUCTION] ** (1/2)

    #Calculate the intensities of the ffts.
    data_intensity = 0.0
    intensities[LOW_PASS] = 0.0
    intensities[BAND_PASS] = 0.0
    intensities[HIGH_PASS] = 0.0
    intensities[GAIN_REDUCTION] = 0.0

    for i in range(0, len(data_fft)):
        data_intensity += data_fft[i]

    for i in range(0, len(fft_data[LOW_PASS])):
        intensities[LOW_PASS] += fft_data[LOW_PASS][i]

    for i in range(0, len(fft_data[BAND_PASS])):
        intensities[BAND_PASS] += fft_data[BAND_PASS][i]

    for i in range(0, len(fft_data[HIGH_PASS])):
        intensities[HIGH_PASS] += fft_data[HIGH_PASS][i]

    for i in range(0, len(fft_data[GAIN_REDUCTION])):
        intensities[GAIN_REDUCTION] += fft_data[GAIN_REDUCTION][i]

    #Calculate time to sleep, but ensure sleeptime isn't negative to not cause an error with time.sleep
    desired_sleep_time = interval - time.time() + starttime
    sleep_time = max(desired_sleep_time, 0)
    time.sleep(sleep_time)
    
    print("Segment: %d\t\t\t\tTime Slept: %f\t\t\t\tSleep Needed: %f\nData Intensity: %f\t\tIdentity (0.5x) Intensity: %f\t\tLow Pass Intensity: %f\nBand Pass Intensity: %f\t\tHigh Pass Intensity: %f\n\n" % (segment, sleep_time, desired_sleep_time, data_intensity, intensities[GAIN_REDUCTION], intensities[LOW_PASS], intensities[BAND_PASS], intensities[HIGH_PASS]), flush=True) #Print at the end when we'd hypothetically do power calculations

#Wait for the signal to finish playing
play_obj.wait_done()