from flask import Flask, render_template, Response, request, jsonify
import server_t
import threading
from multiprocessing import Process
import cv2
import time
import numpy as np

app = Flask(__name__)


#NLPLookUp = {"neutral":0, "anger":1, "contempt":2, "disgust":3, "fear":4, "happy":5, "sadness":6, "surprise":7}

@app.route('/', methods=["GET", "POST"])
def index():
	if request.method == "POST":
		print("post called")
		#counts = np.bincount(server_t.circBuffer, None, 8)
		mybuffer = ""
		counts = [0,0,0,0,0,0,0,0,0]
		allcounts = [0,0,0,0,0,0,0,0,0]
		timeNLP=(0,"none:0", "none")
		if server_t.NLPQueue.qsize() > 0: # new NLP response
			timeNLP = server_t.NLPQueue.get()
			print("got NLP response!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			for item in server_t.circBuffer:
				if abs(int(item[0]) - int(timeNLP[0])) < 3000:
					mybuffer += str(item[1]) + ':'
				counts[item[1]] += 1
				allcounts[item[1]] += 1
		buffersize = max(sum(counts)-counts[8], 1)
		allcountsSize = max(sum(allcounts), 1)
				
		print("my post response:")
		print("useful emotions: " + mybuffer)
		print("latest emotion: " + str(server_t.circBuffer[0][1]))
		print("emotion confidence: " + str(server_t.circBuffer[0][2]) + "%")
		print("speech sentiment: " + timeNLP[1].split(':')[0])
		print("sentence: " + timeNLP[2])
		return jsonify(circBuffer=mybuffer,
					neutral=str(round(allcounts[0]/allcountsSize*100)) + "%", 
					anger=str(round(allcounts[1]/allcountsSize*100)) + "%", 
					contempt=str(round(allcounts[2]/allcountsSize*100)) + "%", 
					disgust=str(round(allcounts[3]/allcountsSize*100)) + "%", 
					fear=str(round(allcounts[4]/allcountsSize*100)) + "%", 
					happy=str(round(allcounts[5]/allcountsSize*100)) + "%", 
					sadness=str(round(allcounts[6]/allcountsSize*100)) + "%", 
					surprise=str(round(allcounts[7]/allcountsSize*100)) + "%",
					
					latest_emo=str(server_t.circBuffer[0][1]),
					emo_conf=str(server_t.circBuffer[0][2]) + "%",
					
					speech_senti=timeNLP[1].split(':')[0],
					speech_conf=timeNLP[1].split(':')[1],
					sentence=timeNLP[2]
					)
		
	return render_template('index.html')


def gen():
	while True:
		start = time.time()
		frame = server_t.currentFrame
		if frame is not None:
			success, encoded_image = cv2.imencode('.jpg', frame)
			content = encoded_image.tobytes()
			yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + content + b'\r\n')
		delay = 0.033 - time.time() - start
		if delay > 0:
			sleep(delay)


@app.route('/video_feed')
def video_feed():
	return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
	

if __name__ == '__main__':
	server = threading.Thread(target=server_t.start_server)
	#server = Process(target=server_t.start_server)
	server.start()
	app.run(host='0.0.0.0')