#REQUIREMENTS:
#pip install scipy
#pip install matplotlib
#pip install librosa
#pip install simpleaudio

from scipy.io import wavfile
from scipy.fft import fft
from scipy import signal
import math
import numpy
import matplotlib.pyplot as pyplot
import simpleaudio
import time
import librosa

#Read in data
data, samplerate = librosa.load('test.wav', sr=8000)
#samplerate, data = wavfile.read('Spoopy.wav')

#If 2 channel audio, take a channel and process it as mono
if(len(data.shape) >= 2 and data.shape[1] == 2):
    temp_data = []
    for i in range(0, len(data)):
        temp_data.append(data[i][0])
    data = numpy.array(temp_data)

rawdata = data #Saving rawdata for playback

bytesPerSample = 2

if(type(data[0]) == numpy.float32):
    #Float32 is 4 bytes in the desired range of -1 to 1
    bytesPerSample = 4
elif(type(data[0]) == numpy.int16):
    #int16 is 2 bytes -32768 to 32767
    data = numpy.array(data)
    data = data.astype(numpy.float32)
    data = data / 32768
elif(type(data[0]) == numpy.int32):
    #int32 is 4 bytes -2147483648 to 2147483647
    data = numpy.array(data)
    data = data.astype(numpy.float32)
    data = data / 2147483648
    bytesPerSample = 4
elif(type(data[0]) == numpy.uint8):
    #uint is 1 byte 0 to 255
    data = numpy.array(data)
    data = data.astype(numpy.float32)
    data = (data / 128) - 1
    bytesPerSample = 1


interval = .05 #Interval is generously set to leave computation time
samplesPerInterval = math.ceil(samplerate * interval) #NOTE: This rounds up, so in instances where samplerate * interval isn't an integer, there may be desync issues, although with conventionally large sampling rates and a clean interval like .1 (aka divide by 10) that shouldn't be a problem.
numSegments = math.ceil(len(data) / samplesPerInterval)

#TODO: Delay play by one interval because that's the delay of the signal processing
play_obj = simpleaudio.play_buffer(rawdata, 1, bytesPerSample, samplerate)


#NOTE: Since samplesPerInterval is dependent on samplerate, with higher sample rate audio, there is a significant performance hit.  Audio should be downsampled, either in code (NYI) or in Audacity, before processing
for segment in range(0,numSegments):
    starttime = time.time()

    #Calculate the ffts of specifically the desired slice of time using some simple indexing
    data_fft = fft(data[(segment * samplesPerInterval):((segment+1) * samplesPerInterval)], samplerate)

    #Multiply by the conjugate element by element to get the power, removing all imaginary components.
    #NOTE: Loops are all separate to allow filters to be completely independent, with an arbitrary amount of filter coefficients
    for i in range(0, len(data_fft)):
        data_fft[i] = data_fft[i] * numpy.conj(data_fft[i])

    #Dividing by the number of samples per interval * samplerate so we can be normalized across sampling rates assuming the same interval
    #PRECONDITION: 8000 / samplerate = 1.  Otherwise multiply data_fft by 8000/samplerate
    data_fft = data_fft / (samplesPerInterval)

    #TEMP: Manually setting min and max as separate variables
    #PRECONDITION: 8Khz sampling rate
    lowpass_min = 0
    lowpass_max = 1000
    bandpass_min = 1000
    bandpass_max = 2000
    highpass_min = 2000
    highpass_max = 4000

    #Calculate the intensities of the ffts.
    data_intensity = 0.0
    lowpass_intensity = 0.0
    bandpass_intensity = 0.0
    highpass_intensity = 0.0

    for i in range(0, len(data_fft)):
        data_intensity += data_fft[i]

    for i in range(lowpass_min, lowpass_max):
        lowpass_intensity += data_fft[i]

    for i in range(bandpass_min, bandpass_max):
        bandpass_intensity += data_fft[i]

    for i in range(highpass_min, highpass_max):
        highpass_intensity += data_fft[i]

    #Calculate time to sleep, but ensure sleeptime isn't negative to not cause an error with time.sleep
    desired_sleep_time = interval - time.time() + starttime
    sleep_time = max(desired_sleep_time, 0)
    time.sleep(sleep_time)
    
    print("Segment: %d\t\tTime Slept: %f\t\tSleep Needed: %f\t\tData Intensity: %f\n" % (segment, sleep_time, desired_sleep_time, data_intensity), flush=True) #Print at the end when we'd hypothetically do power calculations
    print("\t\t\tLow Pass: %f\t\tBand Pass: %f\t\tHigh Pass: %f\n\n" % (lowpass_intensity, bandpass_intensity, highpass_intensity), flush=True)
#Wait for the signal to finish playing
play_obj.wait_done()
print("DATA FFT LEN: ", len(data_fft))