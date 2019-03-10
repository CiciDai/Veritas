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
