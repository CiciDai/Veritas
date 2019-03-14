$(document).ready(function () {    
      var text_pic = {
        "neutral": '"../static/image/neutral.png" alt="neutral"/>',
        "anger": '"../static/image/anger.png" alt="anger"/>',
        "contempt": '"../static/image/contempt.png" alt="contempt"/>',
        "disgust": '"../static/image/disgust.png" alt="disgust"/>',
        "fear": '"../static/image/fear.png" alt="fear"/>',
        "happy": '"../static/image/happy.png" alt="happy"/>',
        "sadness": '"../static/image/sadness.png" alt="sadness"/>',
        "surprise": '"../static/image/surprise.png" alt="surprise"/>',
        "joy": '"../static/image/happy.png" alt="joy"/>',
        "shame": '"../static/image/shame.png" alt="shame"/>',
        "guilt": '"../static/image/guilt.png" alt="guilt"/>'
    }
      
    var nlp_info = {
       "result": "anger",
        "percent": "20%"
    }
    var emotion_info = {
       "neutral": "80%",
        "anger": "30%",
        "contempt": "35%",
        "disgust": '30%',
        "fear": '24%',
        "happy": "15%",
        "sadness": "3%",
        "surprise": '29%'
    }
    
    var emo_dict = {'0':"neutral", '1':"anger", '2':"contempt", '3':"disgust", '4':"fear", '5':"happy", '6':"sadness", '7':"surprise"}
    var nlp_dict = {'anger':0,'disgust':1,'fear':2,'guilt':3,'joy':4,'sadness':5,'shame':6}

    // look up matrix
    m_neutral = [0.6, 0.7, 0.7, 0.5, 0.2, 0.5, 0.5];
    m_anger = [0, 0.3, 0.8, 0.7, 1, 0.5, 0.7];
    m_contempt = [0.2, 0.3, 0.7, 0.8, 0.9, 0.9, 0.8];
    m_disgust = [0.3, 0, 0.6, 0.3, 1, 0.8, 0.3];
    m_fear = [0.8, 0.6, 0, 0.2, 1, 0.3, 0.2];
    m_happy = [1, 1, 1, 1, 0, 1, 1];
    m_sadness=[0.5, 0.5, 0.3, 0, 1, 0, 0];
    m_surprise = [0.5, 0.3, 0.3, 0.5, 0.5, 0.5, 0.3];

    lie_matrix=[m_neutral, m_anger, m_contempt, m_disgust, m_fear, m_happy, m_sadness, m_surprise];

    var nlp_word = {
        "q-content": "Tell me about your weekends.",
        "a-content": "Nothing much. I went to school to do my project."
    }
    
    for(var sen in nlp_word){
        $('#'+ sen).html(nlp_word[sen]);
    }

    var counter = 0;
    // Get overtime data for lying prob
    function updateTimeHist(points){
        var dataPoints = [];
        var font_color = "#90C277";
        var grid_color = "lightgray";
        var options = {
            theme: "light2",
            height: 200,
            axisX: {
                title: "Time (second)",
                titleFontColor: font_color,
                interval: 1,
                gridColor: grid_color,
                gridThickness: 1
            },
            axisY: {
                title: "Probability of Lying",
                titleFontColor: font_color,
                interval: 0.1,
                gridColor: grid_color
            },
            data: [{        
    		type: "spline",       
    		dataPoints: dataPoints
    	}]
        };
        var limit = 30;

        setInterval(function updateData() {
            // if(counter < dataPoints.length){
            //     if(points.length > limit){
            //         points.push(dataPoints[counter]);
            //         points.shift();

            //     }else{
            //         points.push(dataPoints[counter]);
            //     }
            //     counter++;
            //     $("#chartContainer").CanvasJSChart(options);
            // }
            if(counter < points.length){
                if (counter > limit) {
                    dataPoints.push({x: counter, y: points[counter]});
                    dataPoints.shift();
                } else {
                    dataPoints.push({x: counter, y: points[counter]});
                }
                counter++;
                $("#chartContainer").CanvasJSChart(options);
            }
           
        }, 10);
    }    

    // update emotion result
    function updateBar() {
        console.log("update called")
        // update emotion bar
        for (var key in emotion_info) {
            $('#'+key).html("<p>" + emotion_info[key] + "</p>");
            $('#'+key).animate({
                height: (emotion_info[key])
            }, 500);
        }
    }


    function update_values() {
        $SCRIPT_ROOT = "{{ request.script_root|tojson|safe }}";
        $.post('/', {}, 
            function(data) {
                console.log("json is", data);
                emotion_info["neutral"] = data.neutral;
                emotion_info["anger"] = data.anger;
                emotion_info["contempt"] = data.contempt;
                emotion_info["disgust"] = data.disgust;
                emotion_info["fear"] = data.fear;
                emotion_info["happy"] = data.happy;
                emotion_info["sadness"] = data.sadness;
                emotion_info["surprise"] = data.surprise;

                updateBar();

                latest_emo = data.latest_emo;
                emo_conf = data.emo_conf;
                overall["result_emotion"] = emo_dict[latest_emo];
                overall["conf_level"] = emo_conf;

                updateCurrEmo();

                // when speech sentiment is ready
                if (data.speech_senti != "none"){
                    // update response
                    $('#a-content').text(data.sentence);

                    speech_senti = data.speech_senti;
                    speech_conf = data.speech_conf;

                    // update NLP result
                    nlp_info['result'] = speech_senti;
                    nlp_info["percent"] = speech_conf;
                    console.log('nlp_info')
                    updateNLP();

                    // // update time history graph
                    index_j = nlp_dict[speech_senti]
                    circBuffer = data.circBuffer;
                    var points = cvtPoints(circBuffer, index_j, counter);
                    updateTimeHist(points);
                }
            }
        );
    };
  
    var overall = {
        "result_emotion": "neutral",
        "conf_level": "80%"
    }
    
    function updateCurrEmo(){
        console.log("emoticon updated");
        // update conf level and emotion
        for(var result in overall){
            $('#' + result).text(overall[result]);
        }
        // update emoticon
        curr_emo = overall['result_emotion'];
        $('#result_emoji').html('<img id="emoji" src='+ text_pic[curr_emo]);
    }

    function updateNLP(){
        var nlp_emoji = text_pic[nlp_info['result']];
        var nlp_percent = nlp_info["percent"];
        $('#nlp_emoji_div').html(nlp_info['result'] + '<img src =' + nlp_emoji);
        $('#nlp_conf_div').html('confidence level' + "<p id=nlp_conf>" + nlp_percent + "</p>");
    }   


    function cvtPoints(data, index_j, counter){
        var emotions = data.split(':');
        var points = [];
        for(i=0; i<emotions.length; i++){
            var index_i = parseInt(emotions[i]);
            points.push(lie_matrix[index_i][index_j]);
        }
        return points;
    }
     // update nlp result
    // var nlp_emoji = text_pic[nlp_info['result']];
    // var nlp_percent = nlp_info["percent"].toString();
    // $('#nlp_emoji_div').html(nlp_info['result'] + '<img id="nlp_emoji" src=' + nlp_emoji);
    // $('#nlp_conf_div').html('confidence level' + "<p id='nlp_conf'>" + nlp_percent + "</p>");
    
    var mytimer = setInterval(update_values, 500);
    var myupdate = setInterval(updateBar, 500);

    // Initial Values
//    var xValue = 0;
//    var yValue = 0;
//    var newDataCount = 6;
//
//    function addData(data) {
//        if(newDataCount != 1) {
//            $.each(data, function(key, value) {
//                dataPoints.push({x: value[0], y: parseInt(value[1])});
//                xValue++;
//                yValue = parseInt(value[1]);
//            });
//        } else {
//            dataPoints.shift();
//            dataPoints.push({x: data[0][0], y: parseInt(data[0][1])});
//            xValue++;
//            yValue = parseInt(data[0][1]);
//        }
//
//        newDataCount = 1;
//        $("#chartContainer").CanvasJSChart(options);
//        setTimeout(updateData, 1500);
//    }
//    
//    function updateData() {
//        $.getJSON("data.json", addData);
//    }
}); 
