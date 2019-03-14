import io
import socket
import threading
import struct
import time
import queue
import picamera
import cv2
import numpy as np
import PIL.Image
from PIL import ImageTk
from tkinter import *
from multiprocessing import Process, Value

import pyaudio
import audioop
from collections import deque
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

windowWidth = 800
windowHeight = 600
IPAddress = None
root = None

defaultImgPath = PIL.Image.open("./image/veritas.png")
neutralImgPath = PIL.Image.open("./image/neutral.png")
angerImgPath = PIL.Image.open("./image/anger.png")
contemptImgPath = PIL.Image.open("./image/contempt.png")
disgustImgPath = PIL.Image.open("./image/disgust.png")
fearImgPath = PIL.Image.open("./image/fear.png")
happyImgPath = PIL.Image.open("./image/happy.png")
sadnessImgPath = PIL.Image.open("./image/sadness.png")
surpriseImgPath = PIL.Image.open("./image/surprise.png")
unknownImgPath = PIL.Image.open("./image/unknown.png")
emojis = []
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise", "unknown"]
emojiQueue = queue.Queue()

defaultImg = None
newImage = None
videoStream = io.BytesIO()
mutex = threading.Lock()

class VeritasWindow(Frame):
    def __init__(self, master = None):
        Frame.__init__(self, master)
        self.master = master
        self.ipEntry = None
        self.videoPanel = None
        self.emoticonPanel = None
        self.initWindow()
    
    def initWindow(self):
        self.master.title("Veritas")
        self.pack(fill=BOTH, expand=1)
        ipPrompt = Label(self, text="Server IP")
        ipEntry = Entry(self, bd=5)
        self.ipEntry = ipEntry
        ipButton = Button(self, width=5, text="Connect", command=self.getIP)
        quitButton = Button(self, width=5, text="Quit", command=self.exitWindow)
        ipPrompt.pack(side="top", padx=5, pady=5)
        ipEntry.pack(side="top", padx=5, pady=5)
        ipButton.pack(side="top", padx=5, pady=5)
        quitButton.pack(side="top", padx=5, pady=5)

    def exitWindow(self):
        exit()

    def getIP(self):
        global IPAddress
        IPAddress = self.ipEntry.get()
    
    def clearWidgets(self):
        slaveList = self.pack_slaves()
        for slave in slaveList:
            slave.destroy()

    def updateWindow(self):
        self.master.title("Veritas")
        self.pack(fill=BOTH, expand=1)
        defaultImg = ImageTk.PhotoImage(defaultImgPath)
        videoPanel = Label(self, image=defaultImg)
        emoticonPanel = Label(self, image=defaultImg)
        historyPanel = Label(self, height=50, width=500, image=defaultImg)
        labelPanel = Label(self, height=30, image=defaultImg)
        self.videoPanel = videoPanel
        self.emoticonPanel = emoticonPanel
        self.historyPanel = historyPanel
        self.labelPanel = labelPanel
        self.videoPanel.image = defaultImg
        self.emoticonPanel.image = defaultImg
        self.historyPanel.image = defaultImg
        self.labelPanel.image = defaultImg
        videoPanel.pack(side="left", padx=7, pady=5)
        emoticonPanel.pack(side="right", padx=7, pady=5)
        historyPanel.pack(side="bottom", padx=7, pady=5)
        labelPanel.pack(side="bottom", padx=7, pady=5)
        
    def updateWindowGrid(self):
        self.master.title("Veritas")
        defaultImg = ImageTk.PhotoImage(defaultImgPath.resize((480,360)))
        global emojis
        emojis.append(ImageTk.PhotoImage(neutralImgPath.resize((240,240))))
        emojis.append(ImageTk.PhotoImage(angerImgPath.resize((240,240))))
        emojis.append(ImageTk.PhotoImage(contemptImgPath.resize((240,240))))
        emojis.append(ImageTk.PhotoImage(disgustImgPath.resize((240,240))))
        emojis.append(ImageTk.PhotoImage(fearImgPath.resize((240,240))))
        emojis.append(ImageTk.PhotoImage(happyImgPath.resize((240,240))))
        emojis.append(ImageTk.PhotoImage(sadnessImgPath.resize((240,240))))
        emojis.append(ImageTk.PhotoImage(surpriseImgPath.resize((240,240))))
        emojis.append(ImageTk.PhotoImage(unknownImgPath.resize((240,240))))
        
        videoPanel = Label(self, image=defaultImg)
        emoticonPanel = Label(self, text="Emoticons here", image=emojis[0], bg="red")
        confidencePanel = Label(self, text="Confidence", font=("Courier", 35), bg="white")
        historyPanel = Label(self, text="History here", bg="white")
        labelPanel = Label(self, text="Labels here", bg="blue")
        
        self.videoPanel = videoPanel
        self.emoticonPanel = emoticonPanel
        self.confidencePanel = confidencePanel
        self.historyPanel = historyPanel
        self.labelPanel = labelPanel
        
        self.videoPanel.image = defaultImg
        self.emoticonPanel.image = emojis[0]
        self.confidencePanel.image = defaultImg
        
        self.videoPanel.grid(column=0, row=0, columnspan=5, rowspan=5, padx=5, pady=5, sticky=NW)
        self.historyPanel.grid(column=0, row=5, columnspan=7, padx=5, sticky=W)
        self.labelPanel.grid(column=0, row=6, columnspan=7, padx=5, sticky=W)
        self.emoticonPanel.grid(column=5, row=0, columnspan=3, rowspan=3, padx=5, pady=5, sticky='')
        self.confidencePanel.grid(column=5, row=3, padx=5, pady=5, sticky=W)
        
        
    def update(self):
        if not emojiQueue.empty():
            qTuple = emojiQueue.get()
            print(qTuple)
            index = int(qTuple[0])
            self.emoticonPanel.configure(image = emojis[index])
            self.emoticonPanel.image = emojis[index]
            textString = emotions[index]
            if index != 8:
                textString += "\n" + qTuple[1]
            self.confidencePanel.configure(text = textString) 


class SplitFrames(object):
    def __init__(self, connection):
        self.connection = connection
        self.stream = io.BytesIO()
        self.count = 0

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # Start of new frame; send the old one's length
            # then the data
            size = self.stream.tell()
            if size > 0:
                self.connection.write(struct.pack('<L', size))
                self.connection.flush()
                self.stream.seek(0)
                self.connection.write(self.stream.read(size))
                self.count += 1
                #print("sent frame #" + str(self.count) + " at time " + str(time.time()))
                self.stream.seek(0)
        self.stream.write(buf)


def getStrTime():
    msTime = str(round(time.time() * 1000))
    return msTime[-8:]


def startSTT(end):
    print("STT started")
    form_1 = pyaudio.paInt16 # 16-bit resolution
    chans = 1 # 1 channel
    samp_rate = 44100 # 44.1kHz sampling rate
    chunk = 4096 # 2^12 samples for buffer
    record_secs = 3600 # seconds to record
    dev_index = 2 # device index found by p.get_device_info_by_index(ii)

    threshold = 15000
    sliding_window = deque(maxlen=15)
    
    print("init finished")
    
    client = speech.SpeechClient()
    send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    send_socket.connect(('10.18.243.213', 8002))
    
    print("speech client connected")
    
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
        startTime = ""
        # loop through stream and append audio chunks to frame array
        for ii in range(0,int((samp_rate/chunk)*record_secs)):
            data = stream.read(chunk, exception_on_overflow=False)
            predata.append(data)
            rms = audioop.rms(data, 2)
            print(rms)
            if rms > threshold and started is False:
                started = True
                startTime = getStrTime()
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
            myresult = result.alternatives[0].transcript
            print('Transcript: {}'.format(myresult))
            send_socket.send((startTime +";" + myresult).encode())
            print("sent")

        # send result to server
        #send_socket.send((response.results[0].alternatives[0].transcript).encode())
        #print("sent")
        
    # stop the stream, close it, and terminate the pyaudio instantiation
    stream.stop_stream()
    stream.close()
    audio.terminate()

    send_socket.close()


def recv_process(sock):
    print("recv_process running")
    sock.bind(('0.0.0.0', 8001))
    sock.listen(0)
    print("listening on 0")
    #recv_connection = sock.accept()[0].makefile('rb')
    recv_connection = sock.accept()[0]
    print("created recv_connection")
    #response_stream = io.StringIO()
    counter = 0
    resultString = ""
    
    while True:
        print("processing response...")
        recvMessage = recv_connection.recv(1024).decode()
        print("got message " + str(counter) + ": " + recvMessage + " at time " + str(time.time()))
        counter += 1
        if "end_stream" in recvMessage:
            break
        
        resultString += recvMessage
        substrings = resultString.split(":")
        total = len(substrings)
        i = 0
        while i + 2 <= total:
            emojiQueue.put((substrings[i], substrings[i+1]))
            i += 2
        if total % 2 == 0:
            resultString = substrings[total-2] + ":" + substrings[total-1] + ":"
        else:
            resultString = substrings[total-1]
        
    recv_connection.close()
    print("recvconnection closed")


if __name__ == '__main__':
    root = Tk()
    root.geometry(str(windowWidth) + "x" + str(windowHeight))
    app = VeritasWindow(root)

    while IPAddress is None:
        root.update()
    else:
        # Connect a client socket
        print("client connecting on " + IPAddress)
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((IPAddress, 8000))
        print("client connected!")
    
        print("server waiting for connection...")
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_thread = threading.Thread(target=recv_process, args=(server_socket,))
        server_thread.start()
    
        end = Value('i', 0)
        STTThread = threading.Thread(target=startSTT, args=(end,))
        STTThread.start()
        print("started STT process") 
    
        app.clearWidgets()
        app.updateWindowGrid()
    
        # Make a file-like object out of the connection
        send_connection = client_socket.makefile('wb')
        try:
            output = SplitFrames(send_connection)
            with picamera.PiCamera(resolution='VGA', framerate=60) as camera:
                # Start a preview and let the camera warm up for 2 seconds
                camera.hflip = True
                camera.start_preview(fullscreen=False, window=(8,30,480,360))
                time.sleep(2)
    
                start = time.time()
                camera.start_recording(output, format='mjpeg')
                #camera.wait_recording(20)
                while (time.time() - start < 3600):
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    app.update()
                    root.update()
            
                print("out of time!")
                camera.stop_recording()
            
                # Write the terminating 0-length to the connection to let the
                # server know we're done
                send_connection.write(struct.pack('<L', 0))
                send_connection.flush()
                print("wrote end byte to server")
        
            finish = time.time()
            print('Sent %d images in %d seconds at %.2ffps' % (
            output.count, finish-start, output.count / (finish-start)))
        finally:
            print("waiting for server_thread")
            server_thread.join()
            print("server_thread joined")
            server_socket.close()
            print("server socket closed")
            send_connection.close()
            print("send_connection closed")
            client_socket.close()
            print("client socket closed")
            app.exitWindow()
