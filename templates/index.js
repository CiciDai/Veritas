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
        $('#'+key).text(percent);
        $('#'+key).animate({
            height: percent
        }, 1000);
    }
    
    for (var nlp_key in nlp_info) {
        var nlp_percent = nlp_info[nlp_key].toString() + "%";
        $('#n-'+nlp_key).text(nlp_percent);
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
}); 
