$(document).ready(function () {    
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
    
    for (var key in info) {
        $('#'+key).animate({
            height: info[key]
        }, 1000);
        $('#'+key).text(info[key]);
    }
}); 


function update_values() {
            $SCRIPT_ROOT = '{{ request.script_root|tojson|safe }}';
            $.getJSON($SCRIPT_ROOT,
                function(data) {
                    console.log(data)
                    $("#neutral").text(data.neutral+"%")
                    $("#anger").text(data.anger+"%")
                    $("#contempt").text(data.contempt+"%")
                    $("#disgust").text(data.disgust+"%")
                    $("#fear").text(data.fear+"%")
                    $("#happy").text(data.happy+"%")
                    $("#sadness").text(data.sadness+"%")
                    $("#surprise").text(data.surprise+"%")
                });
        }
        
        
setInterval(update_values, 1000)