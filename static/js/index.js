var info = {
    "neutral": "80%",
    "anger": "30%",
    "contempt": "35%",
    "disgust": "30%",
    "fear": "24%",
    "happy": "15%",
    "sadness": "30%",
    "surprise": "29%"
}

var counter = 0

$(document).ready(function () {    
    updateBar()
}); 

function updateBar() {
    console.log("update called")
    // update emotion bar
    for (var key in info) {
        $('#'+key).animate({
            height: (info[key])
        }, 500);
        $('#'+key).append("<p>" + info[key] + "</p>");
    }
    // update nlp bar
    for (var key in info) {
        $('#n-'+key).animate({
            height: (info[key])
        }, 500);
        $('#n-'+key).append("<p>" + info[key] + "</p>");
    }
}


function update_values() {
    $SCRIPT_ROOT = "{{ request.script_root|tojson|safe }}";
    $.post('/', {}, 
        function(data) {
            console.log(data);
            info["neutral"] = data.neutral;
            info["anger"] = data.anger;
            info["contempt"] = data.contempt;
            info["disgust"] = data.disgust;
            info["fear"] = data.fear;
            info["happy"] = data.happy;
            info["sadness"] = data.sadness;
            info["surprise"] = data.surprise;
        }
    );
};

// plot probability over time
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
var counter = 0;
var limit = 15;
function update_graph() {
    if (counter > limit) {
        dataPoints.push({x: counter, y: Math.random()});
        dataPoints.shift();
    } else {
        dataPoints.push({x: counter, y: Math.random()});
    }
    counter++;
    $("#chartContainer").CanvasJSChart(options);
}

var mytimer = setInterval(update_values, 500);
var myupdate = setInterval(updateBar, 500);
var prob_over_time = setInterval(update_graph,1000);
