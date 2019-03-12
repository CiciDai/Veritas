from flask import Flask, render_template, Response, request, jsonify
import server_t
import threading
import cv2
import time
import numpy as np

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
	if request.method == "POST":
		print("post called")
		counts = np.bincount(server_t.circBuffer, None, 8)
		return jsonify(neutral=str(round(counts[0]/30.*100)) + "%", 
					anger=str(round(counts[1]/30.*100)) + "%", 
					contempt=str(round(counts[2]/30.*100)) + "%", 
					disgust=str(round(counts[3]/30.*100)) + "%", 
					fear=str(round(counts[4]/30.*100)) + "%", 
					happy=str(round(counts[5]/30.*100)) + "%", 
					sadness=str(round(counts[6]/30.*100)) + "%", 
					surprise=str(round(counts[7]/30.*100)) + "%"
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
	server.start()
	app.run(host='0.0.0.0')