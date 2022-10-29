$( document ).ready(function() {
	if(!window.WebSocket){
		window.location.replace("error.html");
	} 
});

var ws;
function connect() {
	var username = document.getElementById("usrName").value;
    ws = new WebSocket('ws://localhost:8080/chat');
    
    ws.onmessage = function(data){
    	console.log(data.data);
        printMessage(data.data);
    }
    ws.onopen = function(data){
    	ws.send("#username:"+username);    	
    }
    chatMode();
}

function disconnect() {
    loginMode();
    ws.close();
}

function sendMessage() {
    var message = document.getElementById("message").value;
    ws.send(message);
    document.getElementById("message").value = "";
}






////////////////////////////////
function chatMode(){
    $("#chat").show();
    $("#loginForm").hide();

    var username = document.getElementById("usrName").value;
    $("#welcome").text("Welcome, "+ username);
    document.getElementById("usrName").value = "";
}

function loginMode(){
    $("#chat").hide();
    $("#loginForm").show();
    document.getElementById("textArea").value = "";
    document.getElementById("message").value = "";
}

function printMessage(data){
    var area = document.getElementById("textArea");
    var text = area.value;
    text += "\n" + data;
    area.value = text;
};



