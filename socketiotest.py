from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import threading
import cv2
import time
import numpy as np

app = Flask(__name__)
#app.config['SECRET_KEY'] = 'veritas'
#socketio = SocketIO(app)


@app.route('/')
def index():
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
	

@app.route('/', methods= ['GET'])
def stuff():
	print("dostuff called")
	return jsonify(neutral="10", anger="70", contempt="20", disgust="45", fear="12", happy="60", sadness="5", surprise="75")
	
	
#def updateGraph():
#	counts = np.bincount(server_t.circBuffer)
	
	
if __name__ == '__main__':
	app.run(host='0.0.0.0')
	#socketio.run(app)