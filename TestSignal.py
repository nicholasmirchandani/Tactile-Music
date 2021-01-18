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

#Apply filters, and fix output type
lp1 = [0.127174276079605, 0.0581343489943583, 0.0681122463081755, 0.0766052817881472, 0.0830675938972334, 0.0871853443909994, 0.0884935091352945, 0.0871853443909994, 0.0830675938972334, 0.0766052817881472, 0.0681122463081755, 0.0581343489943583, 0.127174276079605]
bp1 = [0.0109723768383746, -0.0467943943338264, -0.0741398108994016, -0.149777301781025, 0.117993634189359, 0.192388845547486, 0.294512671843853, 0.192388845547486, 0.117993634189359, -0.149777301781025, -0.0741398108994016, -0.0467943943338264, 0.0109723768383746]
hp1 = [-0.0351103427314022, 0.120418583869658, 0.0883153039547716, 0.00865009773730016, -0.134411547496756, -0.277541793649009, 0.662413172546772, -0.277541793649009, -0.134411547496756, 0.00865009773730016, 0.0883153039547716, 0.120418583869658, -0.0351103427314022]
id1 = [0.5]

lp_data = signal.convolve(data, lp1)
bp_data = signal.convolve(data, bp1)
hp_data = signal.convolve(data, hp1)
id_data = signal.convolve(data, id1)

#Fix filtered datatypes to original datatype
lp_data = lp_data.astype(type(data[0]))
bp_data = bp_data.astype(type(data[0]))
hp_data = hp_data.astype(type(data[0]))
id_data = id_data.astype(type(data[0]))

#No longer need to write to output file to ensure nothing broke.
# wavfile.write('output.wav', samplerate, id_data)

bytesPerSample = 2

rawdata = data #Saving rawdata for playback

#Converting data to all be 32 bit floats within -1 to 1, so fft calculations are consistent across filetypes.
lp_data = lp_data.astype(numpy.float32)
bp_data = bp_data.astype(numpy.float32)
hp_data = hp_data.astype(numpy.float32)
id_data = id_data.astype(numpy.float32)

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
    lp_data = lp_data / 32768
    bp_data = bp_data / 32768
    hp_data = hp_data / 32728
    id_data = id_data / 32768
elif(type(data[0]) == numpy.int32):
    #int32 is 4 bytes -2147483648 to 2147483647
    data = numpy.array(data)
    data = data.astype(numpy.float32)
    data = data / 2147483648
    lp_data = lp_data / 2147483648
    bp_data = bp_data / 2147483648
    hp_data = hp_data / 2147483648
    id_data = id_data / 2147483648
    bytesPerSample = 4
elif(type(data[0]) == numpy.uint8):
    #uint is 1 byte 0 to 255
    data = numpy.array(data)
    data = data.astype(numpy.float32)
    data = (data / 128) - 1
    lp_data = (lp_data / 128) - 1
    bp_data = (bp_data / 128) - 1
    hp_data = (hp_data / 128) - 1
    id_data = (id_data / 128) - 1
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
    lp_data_fft = fft(lp_data[(segment * samplesPerInterval):((segment+1) * samplesPerInterval)], samplerate)
    bp_data_fft = fft(bp_data[(segment * samplesPerInterval):((segment+1) * samplesPerInterval)], samplerate)
    hp_data_fft = fft(hp_data[(segment * samplesPerInterval):((segment+1) * samplesPerInterval)], samplerate)
    id_data_fft = fft(id_data[(segment * samplesPerInterval):((segment+1) * samplesPerInterval)], samplerate)

    #Multiply by the conjugate element by element to get the power, removing all imaginary components.
    #NOTE: Loops are all separate to allow filters to be completely independent, with an arbitrary amount of filter coefficients
    for i in range(0, len(data_fft)):
        data_fft[i] = data_fft[i] * numpy.conj(data_fft[i])

    for i in range(0, len(lp_data_fft)):
        lp_data_fft[i] = lp_data_fft[i] * numpy.conj(lp_data_fft[i])

    for i in range(0, len(bp_data_fft)):
        bp_data_fft[i] = bp_data_fft[i] * numpy.conj(bp_data_fft[i])

    for i in range(0, len(hp_data_fft)):
        hp_data_fft[i] = hp_data_fft[i] * numpy.conj(hp_data_fft[i])

    for i in range(0, len(id_data_fft)):
        id_data_fft[i] = id_data_fft[i] * numpy.conj(id_data_fft[i])

    #Dividing by the length of the field so we can be normalized across datasets
    data_fft = data_fft / (len(data))
    lp_data_fft = lp_data_fft / (len(lp_data))
    bp_data_fft = bp_data_fft / (len(bp_data))
    hp_data_fft = hp_data_fft / (len(hp_data))
    id_data_fft = id_data_fft / (len(id_data))

    #Taking the square root of ffts so squaring of power to remove imaginary components doesn't break further math
    #Necessary so a 2x gain reduction is a 2x intensity reduction
    data_fft = data_fft ** (1/2)
    lp_data_fft = lp_data_fft ** (1/2)
    bp_data_fft = bp_data_fft ** (1/2)
    hp_data_fft = hp_data_fft ** (1/2)
    id_data_fft = id_data_fft ** (1/2)
    
    #Using math.log10 because np.log10 was giving errors
    #TODO: Is this necessary now that we're no longer plotting?  Removing for now because it's a massive time sink
    '''
    data_decibel = []
    lp_data_decibel = []
    bp_data_decibel = []
    hp_data_decibel = []
    id_data_decibel = []
    for i in range(0, len(data_fft)):
        data_fft[i] = numpy.real(data_fft[i])
        data_decibel.append(10 * math.log10(data_fft[i]))
    for i in range(0, len(lp_data_fft)):
        lp_data_fft[i] = numpy.real(lp_data_fft[i])
        lp_data_decibel.append(10 * math.log10(lp_data_fft[i]))
    for i in range(0, len(bp_data_fft)):
        bp_data_fft[i] = numpy.real(bp_data_fft[i])
        bp_data_decibel.append(10 * math.log10(bp_data_fft[i]))
    for i in range(0, len(hp_data_fft)):
        hp_data_fft[i] = numpy.real(hp_data_fft[i])
        hp_data_decibel.append(10 * math.log10(hp_data_fft[i]))
    for i in range(0, len(id_data_fft)):
        id_data_fft[i] = numpy.real(id_data_fft[i])
        id_data_decibel.append(10 * math.log10(id_data_fft[i]))
    '''

    #Calculate the intensities of the ffts.
    data_intensity = 0.0
    lp_data_intensity = 0.0
    bp_data_intensity = 0.0
    hp_data_intensity = 0.0
    id_data_intensity = 0.0

    for i in range(0, len(data_fft)):
        data_intensity += data_fft[i]

    for i in range(0, len(lp_data_fft)):
        lp_data_intensity += lp_data_fft[i]

    for i in range(0, len(bp_data_fft)):
        bp_data_intensity += bp_data_fft[i]

    for i in range(0, len(hp_data_fft)):
        hp_data_intensity += hp_data_fft[i]

    for i in range(0, len(id_data_fft)):
        id_data_intensity += id_data_fft[i]

    #Calculate time to sleep, but ensure sleeptime isn't negative to not cause an error with time.sleep
    desired_sleep_time = interval - time.time() + starttime
    sleep_time = max(desired_sleep_time, 0)
    time.sleep(sleep_time)
    
    print("Segment: %d\t\t\t\tTime Slept: %f\t\t\t\tSleep Needed: %f\nData Intensity: %f\t\tIdentity (0.5x) Intensity: %f\t\tLow Pass Intensity: %f\nBand Pass Intensity: %f\t\tHigh Pass Intensity: %f\n\n" % (segment, sleep_time, desired_sleep_time, data_intensity, id_data_intensity, lp_data_intensity, bp_data_intensity, hp_data_intensity), flush=True) #Print at the end when we'd hypothetically do power calculations

    #No longer plotting results to verify findings because there's so many individual time slices
    '''
    minVal = 9999
    maxVal = -9999
    for i in range(0, len(id_data)):
        if(id_data[i] > maxVal):
            maxVal = id_data[i]
        if(id_data[i] < minVal):
            minVal = id_data[i]

    #print("NEW MIN: ", minVal, "\nNEW MAX: ", maxVal)
    #Plot Results!
    fig, axs = pyplot.subplots(2, 2)
    axs[0,0].plot(data)
    axs[0,0].set_title('Old numbers')

    axs[0,1].plot(id_data)
    axs[0,1].set_title('New numbers')

    #setting min and max so we have some bounds to relatively compare everything
    ymin = -50
    ymax = 50

    #Using samplerate/2 because Nyquist rate only ensures everything up to that point is accurate
    axs[1,0].plot(data_decibel[0:(int)(samplerate/2)])
    axs[1,0].set_title('Input Loudness')
    axs[1,0].set_ylim(pyplot.ylim(bottom=ymin, top=ymax))

    axs[1,1].plot(id_data_decibel[0:(int)(samplerate/2)])
    axs[1,1].set_title('Output Loudness')
    axs[1,1].set_ylim(pyplot.ylim(bottom=ymin, top=ymax))

    
    pyplot.figure(2)
    data_decibel = numpy.array(data_decibel)
    id_data_decibel = numpy.array(id_data_decibel)
    #pyplot.plot(id_data_decibel[0:(int)(samplerate/2)]-data_decibel[0:(int)(samplerate/2)])
    pyplot.plot(id_data_fft[0:(int)(samplerate/2)]/data_fft[0:(int)(samplerate/2)])
    pyplot.gca().set_ylim([0,10])

    pyplot.show()
    '''

#Wait for the signal to finish playing
play_obj.wait_done()