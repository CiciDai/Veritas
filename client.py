import io
import socket
import struct
import time
import picamera
import cv2
import numpy as np
import PIL.Image
from PIL import ImageTk
from tkinter import *

windowWidth = 640
windowHeight = 480
IPAddress = None
root = None
defaultImgPath = PIL.Image.open("veritas.png")
defaultImg = None
newImage = None
videoStream = io.BytesIO()


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
        videoPanel = Label(self, width=300, image=defaultImg)
        emoticonPanel = Label(self, width=300, image=defaultImg)
        self.videoPanel = videoPanel
        self.emoticonPanel = emoticonPanel
        self.videoPanel.image = defaultImg
        self.emoticonPanel.image = defaultImg
        videoPanel.pack(side="left", padx=7, pady=5)
        emoticonPanel.pack(side="left", padx=7, pady=5)
        
    def update(self):
        if videoStream.tell() > 0:
            #nextImg = ImageTk.PhotoImage(newImage)
            nextImg = ImageTk.PhotoImage(PIL.Image.open(videoStream))
            self.videoPanel.configure(image=nextImg)
            #self.videoPanel.image = ImageTk.PhotoImage(PIL.Image.open(videoStream))
            #self.panel.image = ImageTk.PhotoImage(newImage)
            self.videoPanel.image = nextImg
            videoStream.seek(0)
            videoStream.truncate()
            print("got here")


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
                print("wrote once")
                #img = np.asarray(PIL.Image.open(self.stream))
                #global newImg
                #newImg = PIL.Image.open(self.stream)
                videoStream.seek(0)
                videoStream.write(self.stream.read(size))
                #cv2.imshow("image", img)
                #cv2.waitKey(1)
                self.stream.seek(0)
        self.stream.write(buf)

root = Tk()
root.geometry(str(windowWidth) + "x" + str(windowHeight))
app = VeritasWindow(root)
# root.mainloop()

while IPAddress is None:
    root.update()
else:
    # Connect a client socket to my_server:8000 (change my_server to the
    # hostname of your server)
    print("connecting on " + IPAddress)
    client_socket = socket.socket()
    client_socket.connect((IPAddress, 8000))
    print("connected!")
    
    app.clearWidgets()
    app.updateWindow()
    
    # Make a file-like object out of the connection
    connection = client_socket.makefile('wb')
    try:
        output = SplitFrames(connection)
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
            
            camera.stop_recording()
        
            # Write the terminating 0-length to the connection to let the
            # server know we're done
            connection.write(struct.pack('<L', 0))
    finally:
        connection.close()
        client_socket.close()
        finish = time.time()
    print('Sent %d images in %d seconds at %.2ffps' % (
        output.count, finish-start, output.count / (finish-start)))
