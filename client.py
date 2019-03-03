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
        defaultImg = ImageTk.PhotoImage(defaultImgPath)
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
        confidencePanel = Label(self, text="Confidence", bg="green")
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
        mutex.acquire()
        if videoStream.tell() > 0:
            #nextImg = ImageTk.PhotoImage(newImage)
            nextImg = ImageTk.PhotoImage(PIL.Image.open(videoStream).resize((480,360)))
            videoStream.seek(0)
            videoStream.truncate()
            self.videoPanel.configure(image=nextImg)
            #self.videoPanel.image = ImageTk.PhotoImage(PIL.Image.open(videoStream))
            #self.panel.image = ImageTk.PhotoImage(newImage)
            self.videoPanel.image = nextImg
        mutex.release()
        if not emojiQueue.empty():
            qTuple = emojiQueue.get()
            index = int(qTuple[0])
            self.emoticonPanel.configure(image = emojis[index])
            self.emoticonPanel.image = emojis[index]
            textString = emotions[index]
            if index != 8:
                textString += " with " + qTuple[1] + " confidence"
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
                self.stream.seek(0)
                #img = np.asarray(PIL.Image.open(self.stream))
                #global newImg
                #newImg = PIL.Image.open(self.stream)
                mutex.acquire()
                videoStream.seek(0)
                videoStream.write(self.stream.read(size))
                mutex.release()
                #cv2.imshow("image", img)
                #cv2.waitKey(1)
                self.stream.seek(0)
        self.stream.write(buf)


def recv_process(sock):
    print("recv_process running")
    sock.bind(('0.0.0.0', 8001))
    sock.listen(0)
    print("listening on 0")
    #recv_connection = sock.accept()[0].makefile('rb')
    recv_connection = sock.accept()[0]
    print("created recv_connection")
    #response_stream = io.StringIO()
    
    while True:
        print("starting loop")
        #response_len = struct.unpack('<L', recv_connection.read(struct.calcsize('<L')))[0]
        #if not response_len:
        #    print("invalid response_len, breaking")
        #    break
        print("processing response...")
        recvMessage = recv_connection.recv(128).decode()
        print(recvMessage)
        if recvMessage == "end_stream":
            break
        if recvMessage == "Unknown":
            emojiQueue.put(("8", "Unknown"))
        else:
            substrings = recvMessage.split(':')
            emojiQueue.put((substrings[0], substrings[1]))
        
        #response_stream.write(recv_connection.read(response_len))
        #response_stream.seek(0)
        #print(response_stream.read())
        #response_stream.seek(0)
    recv_connection.close()
    print("recvconnection closed")


root = Tk()
root.geometry(str(windowWidth) + "x" + str(windowHeight))
app = VeritasWindow(root)
# root.mainloop()

while IPAddress is None:
    root.update()
else:
    # Connect a client socket to my_server:8000 (change my_server to the
    # hostname of your server)
    print("client connecting on " + IPAddress)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((IPAddress, 8000))
    print("client connected!")
    
    print("server waiting for connection...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_thread = threading.Thread(target=recv_process, args=(server_socket,))
    server_thread.start()
    #server_socket.bind(('0.0.0.0', 8001))
    #server_socket.listen(0)
    #connection = server_socket.accept()[0].makefile('rb')
    #response_stream = io.StringIO()
    
    #response_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
    #if not response_len:
    #    break
    #
    #response_stream.write(connection.read(response_len))
    #response_stream.seek(0)
    
    
    
    
    
    
    app.clearWidgets()
    app.updateWindowGrid()
    
    # Make a file-like object out of the connection
    send_connection = client_socket.makefile('wb')
    try:
        output = SplitFrames(send_connection)
        with picamera.PiCamera(resolution='VGA', framerate=90) as camera:
            # Start a preview and let the camera warm up for 2 seconds
            #camera.start_preview()
            time.sleep(2)
    
            start = time.time()
            camera.start_recording(output, format='mjpeg')
            #camera.wait_recording(20)
            while (time.time() - start < 10):
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
