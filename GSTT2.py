import pyaudio
import wave
import audioop
import socket
from collections import deque
from multiprocessing import Value

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types


def startSTT(end):
    form_1 = pyaudio.paInt16 # 16-bit resolution
    chans = 1 # 1 channel
    samp_rate = 44100 # 44.1kHz sampling rate
    chunk = 4096 # 2^12 samples for buffer
    record_secs = 3600 # seconds to record
    dev_index = 2 # device index found by p.get_device_info_by_index(ii)

    threshold = 15000
    sliding_window = deque(maxlen=15)

    client = speech.SpeechClient()
    #send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #send_socket.connect(('10.0.0.194', 8002))

    audio = pyaudio.PyAudio() # create pyaudio instantiation
    
    # create pyaudio stream
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                    input_device_index = dev_index, input = True, \
                    frames_per_buffer=chunk)
    print("recording")
    
    predata = deque(maxlen=10)

    while end.value == 0:
        frames = []
        started = False
        # loop through stream and append audio chunks to frame array
        for ii in range(0,int((samp_rate/chunk)*record_secs)):
            data = stream.read(chunk, exception_on_overflow=False)
            predata.append(data)
            rms = audioop.rms(data, 2)
            print(rms)
            if rms > threshold and started is False:
                started = True
                print('started')
            if started:
                frames.append(data)
                sliding_window.append(rms)
                if sum(ii < threshold for ii in sliding_window) >= 15:
                    print("ending")
                    break

        print("finished recording")
        
        for i in range(len(predata)):
            print('added a frame')
            frames.insert(0, predata.pop())

        audio = types.RecognitionAudio(content=b''.join(frames))
        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code='en-US')

        response = client.recognize(config, audio)

        for result in response.results:
            print('Transcript: {}'.format(result.alternatives[0].transcript))

        # send result to server
        #send_socket.send((result.alternatives[0].transcript).encode())
        print("sent")
        
        
    # stop the stream, close it, and terminate the pyaudio instantiation
    stream.stop_stream()
    stream.close()
    audio.terminate()

    #send_socket.close()


if __name__ == '__main__':
    end = Value('i', 0)
    startSTT(end)
# save the audio frames as .wav file
#wavefile = wave.open(wav_output_filename,'wb')
#wavefile.setnchannels(chans)
#wavefile.setsampwidth(audio.get_sample_size(form_1))
#wavefile.setframerate(samp_rate)
#wavefile.writeframes(b''.join(frames))
#wavefile.close()