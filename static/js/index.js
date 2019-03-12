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
    for (var key in info) {
        $('#'+key).animate({
            height: (info[key])
        }, 500);
        $('#'+key).text(info[key]);
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


var mytimer = setInterval(update_values, 500);
var myupdate = setInterval(updateBar, 500);