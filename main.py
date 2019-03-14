from flask import Flask, render_template, jsonify, request
import numpy as np
# import server_t
import time

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print("post called")


        bufferSize = 5.0
        buffer = [0,1,2,4,5]
        counts = np.bincount(buffer, None, 8)
        passBuffer = "0:1:2:4:5:6:7:3:1:6:2:1:3:5"
        # server_t need to send circBuffer, emo_conf, latest_emo, speech_senti, sentence, speech_conf
        # latest_emo and circBuffer in #, emo_conf and speech_conf in %, speech_senti in label (none if no result)

        return jsonify(circBuffer=str(passBuffer),
                       neutral=str(round(counts[0] / bufferSize * 100)) + "%",
                       anger=str(round(counts[1] / bufferSize * 100)) + "%",
                       contempt=str(round(counts[2] / bufferSize * 100)) + "%",
                       disgust=str(round(counts[3] / bufferSize * 100)) + "%",
                       fear=str(round(counts[4] / bufferSize * 100)) + "%",
                       happy=str(round(counts[5] / bufferSize * 100)) + "%",
                       sadness=str(round(counts[6] / bufferSize * 100)) + "%",
                       surprise=str(round(counts[7] / bufferSize * 100)) + "%",

                       # latest_emo=str(server_t.latest_emo),
                       # emo_conf=str(round(server_t.emo_conf * 100)) + "%",
                       #
                       # speech_senti=str(server_t.speech_senti),
                       # speech_conf=str(server_t.speech_conf),
                       # sentence=str(server_t.sentence),

                       latest_emo=str(1),
                       emo_conf=str(30) + "%",

                       speech_senti=str('joy'),
                       speech_conf=str('30%'),
                       sentence=str('finally done')
                       )

    return render_template('index.html')


if __name__ == '__main__':

    app.run(host='127.0.0.1', port=8081, debug=True)
