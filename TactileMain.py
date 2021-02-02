#BleakTest.py is the code from a communicating computer.  Is some simplified code to test that main.py is correctly responding to communications.
#REQUIREMENTS:
#pip install bleak
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
from bleak import BleakClient
import bleak
import asyncio

TARGET_UUID = 'b7328f9c-c89e-4d74-9a5e-000000000000'
UART_TX = 'b7328f9c-c89e-4d74-9a5e-000000000001' #UART'S TX is Bleak's RX
UART_RX = 'b7328f9c-c89e-4d74-9a5e-000000000002' #UART'S RX is Bleak's TX

async def play_file(filename):
    TARGET_UUID = 'b7328f9c-c89e-4d74-9a5e-000000000000'
    UART_TX = 'b7328f9c-c89e-4d74-9a5e-000000000001' #UART'S TX is Bleak's RX
    UART_RX = 'b7328f9c-c89e-4d74-9a5e-000000000002' #UART'S RX is Bleak's TX

    address = "A4:CF:12:58:72:2A"  #NOTE: MAC address is per device, so this needs to be changed
    client = BleakClient(address)

    await client.connect()

    read_string = b''
    read_string = await client.read_gatt_char(UART_TX)
    print("TEST STRING: ", read_string.decode('UTF-8'))

    #Read in data
    samplerate, data = wavfile.read(filename)
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

    bytesPerSample = 2

    #Apply filters, and fix output type
    filters[LOW_PASS] = [0.127174276079605, 0.0581343489943583, 0.0681122463081755, 0.0766052817881472, 0.0830675938972334, 0.0871853443909994, 0.0884935091352945, 0.0871853443909994, 0.0830675938972334, 0.0766052817881472, 0.0681122463081755, 0.0581343489943583, 0.127174276079605]
    filters[BAND_PASS] = [0.0109723768383746, -0.0467943943338264, -0.0741398108994016, -0.149777301781025, 0.117993634189359, 0.192388845547486, 0.294512671843853, 0.192388845547486, 0.117993634189359, -0.149777301781025, -0.0741398108994016, -0.0467943943338264, 0.0109723768383746]
    filters[HIGH_PASS] = [-0.0351103427314022, 0.120418583869658, 0.0883153039547716, 0.00865009773730016, -0.134411547496756, -0.277541793649009, 0.662413172546772, -0.277541793649009, -0.134411547496756, 0.00865009773730016, 0.0883153039547716, 0.120418583869658, -0.0351103427314022]
    filters[GAIN_REDUCTION] = [0.5]

    for i in range(0, NUM_FILTERS):
        filtered_data[i] = signal.convolve(data, filters[i])
        #Converting data to all be 32 bit floats within -1 to 1, so fft calculations are consistent across filetypes.
        filtered_data[i] = filtered_data[i].astype(numpy.float32)


    if(type(data[0]) == numpy.float32):
        #Float32 is 4 bytes in the desired range of -1 to 1
        bytesPerSample = 4
    elif(type(data[0]) == numpy.int16):
        #int16 is 2 bytes -32768 to 32767
        data = numpy.array(data)
        data = data.astype(numpy.float32)
        data = data / 32768
        for i in range(0, NUM_FILTERS):
            filtered_data[i] = filtered_data[i] / 32768
    elif(type(data[0]) == numpy.int32):
        #int32 is 4 bytes -2147483648 to 2147483647
        data = numpy.array(data)
        data = data.astype(numpy.float32)
        data = data / 2147483648
        for i in range(0, NUM_FILTERS):
            filtered_data[i] = filtered_data[i] / 2147483648
        bytesPerSample = 4
    elif(type(data[0]) == numpy.uint8):
        #uint is 1 byte 0 to 255
        data = numpy.array(data)
        data = data.astype(numpy.float32)
        data = (data / 128) - 1
        for i in range(0, NUM_FILTERS):
            filtered_data[i] = (filtered_data[i] / 128) - 1
        bytesPerSample = 1

    interval = .2 #Interval is .2 sec to start to leave computation time
    samplesPerInterval = math.ceil(samplerate * interval) #NOTE: This rounds up, so in instances where samplerate * interval isn't an integer, there may be desync issues, although with conventionally large sampling rates and a clean interval like .1 (aka divide by 10) that shouldn't be a problem.
    numSegments = math.ceil(len(data) / samplesPerInterval)

    #TODO: Delay play by one interval because that's the delay of the signal processing
    play_obj = simpleaudio.play_buffer(rawdata, 1, bytesPerSample, samplerate)

    #NOTE: Since samplesPerInterval is dependent on samplerate, with higher sample rate audio, there is a significant performance hit.  Audio should be downsampled, either in code (NYI) or in Audacity, before processing
    for segment in range(0,numSegments):
        starttime = time.time()

        #Calculate the ffts of specifically the desired slice of time using some simple indexing
        data_fft = fft(data[(segment * samplesPerInterval):((segment+1) * samplesPerInterval)], samplerate)
        for i in range(0, NUM_FILTERS):
            fft_data[i] = fft(filtered_data[i][(segment * samplesPerInterval):((segment+1) * samplesPerInterval)], samplerate)

        #Multiply by the conjugate element by element to get the power, removing all imaginary components.
        #NOTE: Loops are all separate to allow filters to be completely independent, with an arbitrary amount of filter coefficients
        for i in range(0, len(data_fft)):
            data_fft[i] = data_fft[i] * numpy.conj(data_fft[i])

        for i in range(0, NUM_FILTERS):
            for j in range(0, len(fft_data[i])):
                fft_data[i][j] = fft_data[i][j] * numpy.conj(fft_data[i][j])

        #Dividing by the length of the field so we can be normalized across datasets
        data_fft = data_fft / (len(data))
        for i in range(0, NUM_FILTERS):
            fft_data[i] = fft_data[i] / len(filtered_data[i])

        #Taking the square root of ffts so squaring of power to remove imaginary components doesn't break further math
        #Necessary so a 2x gain reduction is a 2x intensity reduction
        data_fft = data_fft ** (1/2)
        for i in range(0, NUM_FILTERS):
            fft_data[i] = fft_data[i] ** (1/2)

        #Calculate the intensities of the ffts.
        data_intensity = 0.0
        for i in range(0, NUM_FILTERS):
            intensities[i] = 0.0

        for i in range(0, len(data_fft)):
            data_intensity += data_fft[i]

        for i in range(0, NUM_FILTERS):
            for j in range(0, len(fft_data[i])):
                intensities[i] += fft_data[i][j]

        write_bytes = b''
        for i in range(0, NUM_FILTERS):
            #TODO: Instead of writing the intensity to bytes write something more insightful
            if(intensities[i] > 100):
                modulated_intensity = 1023
            elif(intensities[i] > 65):
                modulated_intensity = 512
            else:
                modulated_intensity = 0
            write_bytes += modulated_intensity.to_bytes(2, 'big')
        await client.write_gatt_char(UART_RX, write_bytes)

        #Calculate time to sleep, but ensure sleeptime isn't negative to not cause an error with time.sleep
        desired_sleep_time = interval - time.time() + starttime
        sleep_time = max(desired_sleep_time, 0)
        time.sleep(sleep_time)
        
        print("Segment: %d\t\t\t\tTime Slept: %f\t\t\t\tSleep Needed: %f\nData Intensity: %f\t\tIdentity (0.5x) Intensity: %f\t\tLow Pass Intensity: %f\nBand Pass Intensity: %f\t\tHigh Pass Intensity: %f\n\n" % (segment, sleep_time, desired_sleep_time, data_intensity, intensities[GAIN_REDUCTION], intensities[LOW_PASS], intensities[BAND_PASS], intensities[HIGH_PASS]), flush=True) #Print at the end when we'd hypothetically do power calculations

    #Wait for the signal to finish playing
    play_obj.wait_done()


asyncio.run(play_file("test.wav"))
