from flask import Flask, render_template, Response
import server_t
import threading
import cv2
import time

app = Flask(__name__)


@app.route('/')
def index():
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