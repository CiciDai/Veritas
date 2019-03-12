from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import cv2
import time
import numpy as np

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
	if request.method == "POST":
		print("post called")
		return jsonify(neutral=genPerc(), anger=genPerc(), contempt=genPerc(), disgust=genPerc(), fear=genPerc(), happy=genPerc(), sadness=genPerc(), surprise=genPerc())
		
	return render_template('index.html')

	
def gen():
	while True:
		start = time.time()
		#frame = server_t.currentFrame
		frame = None
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
	app.run(host='0.0.0.0')