$(document).ready(function () {    
    var nlp_info = {
       "neutral": 80,
        "anger": 30,
        "contempt": 35,
        "disgust": 30,
        "fear": 24,
        "happy": 15,
        "sadness": 3,
        "surprise": 29
    }
    var emotion_info = {
       "neutral": 80,
        "anger": 30,
        "contempt": 35,
        "disgust": 30,
        "fear": 24,
        "happy": 15,
        "sadness": 3,
        "surprise": 29
    }
    
    for (var key in emotion_info) {
        var percent = emotion_info[key].toString() +"%";
        $('#'+key).append("<p>" + percent + "</p>");
        $('#'+key).animate({
            height: percent
        }, 1000);
    }
    
    for (var nlp_key in nlp_info) {
        var nlp_percent = nlp_info[nlp_key].toString() + "%";
        $('#n-'+nlp_key).append("<p>" + nlp_percent + "</p>");
        $('#n-'+nlp_key).animate({
            height: nlp_percent
        }, 1000);
    }
    var nlp_word = {
        "q-content": "Tell me about your weekends.",
        "a-content": "Nothing much. I went to school to do my project."
    }
    
    for(var sen in nlp_word){
        $('#'+ sen).text(nlp_word[sen]);
    }
    
    var overall = {
        "result_emotion": "neutral",
        "conf_level": "80%"
    }
    
    for(var result in overall){
        $('#' + result).text(overall[result]);
    }
    
    var text_pic = {
        "neutral": '"./image/neutral.png" alt="neutral"/>',
        "anger": '"./image/anger.png" alt="anger"/>',
        "contempt": '"./image/contempt.png" alt="contempt"/>',
        "disgust": '"./image/disgust.png" alt="disgust"/>',
        "fear": '"./image/fear.png" alt="fear"/>',
        "happy": '"./image/happy.png" alt="happy"/>',
        "sadness": '"./image/sadness.png" alt="sadness"/>',
        "surprise": '"./image/surprise.png" alt="surprise"/>'
    }
    
    var emotion = $('#result_emotion').text();
    $('#result_emoji').prepend('<img id="emoji" src=' + text_pic[emotion]);
    
// Get overtime data for lying prob
    var dataPoints = [];
    
    var font_color = "#90C277";
    var grid_color = "lightgray";
    var options = {
//        animationEnabled: true,
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
    var counter = 0;
    var limit = 15;
    setInterval(function updateData() {
        if (counter > limit) {
            dataPoints.push({x: counter, y: Math.random()});
            dataPoints.shift();
        } else {
            dataPoints.push({x: counter, y: Math.random()});
        }
        counter++;
        $("#chartContainer").CanvasJSChart(options);
    }, 1000);
    
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
