var eventFlag = [];
var clickedButton = [];
var userID = document.getElementById("userID").value;
var sources = [{url: '/calendar/'.concat(userID),color : "blue"}];

$(document).ready(function () {
    var currentDate = document.getElementById("currentDate").value;
    $('#calendar').fullCalendar({
        aspectRatio: 2,
        themeSystem: "bootstrap4",
        eventSources: sources,
        header: {
            left: 'prev,next today',
            center: 'title',
            right: 'month,agendaWeek,agendaDay'
        },
        firstDay: 1,
        defaultDate: currentDate,
        defaultView: 'agendaWeek',
        editable: true,
        slotLabelFormat: "HH:mm",
        timeFormat: 'H(:mm)',
        selectable: true,
        eventLimit: true,
        selectHelper: true,
        navLinks: true,
        eventClick: function (event) {
            console.log(event);
            clickedButton = event;
            document.getElementById("beginField").value = event.start.format("YYYY/MM/DD HH:mm");
            document.getElementById("endField").value = event.end.format("YYYY/MM/DD HH:mm");
            document.getElementById("titleField").value = event.title;
            $("#deleteButton").show();
            if (event.repeatInterval != null) {
                document.getElementById("pauseFrom").value = event.pauseFrom === null ? "" : moment(event.pauseFrom).format("YYYY/MM/DD hh:mm");
                document.getElementById("pauseTo").value = event.pauseTo === null ? "" : moment(event.pauseTo).format("YYYY/MM/DD hh:mm");
                document.getElementById("dropdownMenuLink").innerText = event.repeatInterval;
                $("#modalHeader").html("<b>Update  event: " + event.title + "<b>");
            }
            else {
                $("#modalHeader").html("<b>Update event: " + event.title + "<b>");
            }
            $('#myModal').modal();
            eventFlag = "UPDATE";
        },
        select: function (start, end) {
            start = start.format("YYYY/MM/DD HH:mm");
            end = end.format("YYYY/MM/DD HH:mm");
            document.getElementById('beginField').value = start;
            document.getElementById('endField').value = end;
            document.getElementById("titleField").value = "New event";
            $("#deleteButton").hide();
            $("#modalHeader").html("<b>Create new event<b>");
            $('#myModal').modal();
            eventFlag = "INSERT";
        }
        /*,eventMouseover: function (event, jsEvent, view) {
            console.log(event);
            let result = $('#aa')
            let a = `<div className="container">
                <h3>Popover Example</h3>
                <a href="#" data-toggle="popover" title="Popover Header" data-content="Some content inside the popover">Toggle
                    popover</a>
            </div>`;
            result.html(a);
        }*/
    });

});

$(document).ready(function () {
    $("#confirmAction").click(function () {
        $('#warningModal').modal('toggle');
        $('#myModal').modal('toggle');
    });

    $("#closeAction").click(function () {
        $('#warningModal').modal('toggle');
    });
});

function posts(us,name){
    for (var j = 0; j < sources.length; ++j){
        var url_id = sources[j].url.split("/")[2];
        console.log(url_id);
        if(us === url_id || us === userID){
            $("#dropdownActivityMap").hide();
            return;
        }
    }

   $("#calendar").fullCalendar('addEventSource','/calendar/' + us);
    $("#dropdownActivityMap").hide();
    document.getElementById("search-boxActivityMap").value = "";

    var newButton = document.createElement("button");
    newButton.innerText=name;
    newButton.className="btn btn-primary";
    newButton.setAttribute("id",name);
    newButton.setAttribute("onclick", "deleteBtn("+name+")");
    newButton.setAttribute("value", us);
    document.getElementById("buttonGroup").append(newButton);

    sources.push({url : '/calendar/'.concat(us), color : getRandomColor()});


}

$(document).ready(function () {
    $('#search-boxActivityMap').on('keyup', _.debounce(function (e) {
        $("#dropdownActivityMap").show();
        let query = $('#search-boxActivityMap').val();
        let result = $('#dropdownActivityMap');

        if (!query.trim().length) {
            result.html("");
            return;
        }

        console.debug(query);

        $.ajax({
            type: "GET",
            url: "/search",
            data: {query: query},
            success: function (data) {
                result.html("");

                let users = data.users;
                let groups = data.groups;

                let newHtml = '<ul class="list-group">';
                for (let i = 0; i < users.length; i++) {
                    newHtml += `<a id=x" class="list-group-item list-group-item-action" onclick="posts('${users[i].id}','${users[i].username}')" >
                                            <div class="row mb-1 mt-1" >
                                                <div class="col-auto">
                                                    <img class="user-img"" src="../img/default_profile_picture.jpg">
                                                </div>
                                                <div class="col-auto">
                                                    <div>${users[i].firstName} ${users[i].lastName}</div>
                                                </div>
                                            </div>
                                        </a>`;
                }
                newHtml += '</ul>';

                result.html(newHtml);
            }
        });
    }, 200));
});

function beginDateTimePicker() {
    $(document).ready(function () {
        $('#beginField').datetimepicker();
    });
}

function endDateTimePicker() {
    $(document).ready(function () {
        $('#endField').datetimepicker();
    });
}

function pauseFromDateTimePicker() {
    $(document).ready(function () {
        $('#pauseFrom').datetimepicker();
    });

}

function pauseToDateTimePicker() {
    $(document).ready(function () {
        $('#pauseTo').datetimepicker();
    });
}

function saveOptions() {
    if (hasErrors()) {
        return;
    }
    var calChanged = true;
    document.getElementById("routineTypeField").value = document.getElementById("dropdownMenuLink").innerText;
    if (eventFlag === "UPDATE") {
        document.getElementById("httpMetoda").value = "put";
        document.getElementById("eventId").value = clickedButton.id;
        var beginChange = document.getElementById('beginField').value === clickedButton.start.format("YYYY/MM/DD hh:mm");
        var endChange = document.getElementById('endField').value === clickedButton.end.format("YYYY/MM/DD hh:mm");
        var titleChange = document.getElementById('titleField').value === clickedButton.title;
        var pauseFromChange = document.getElementById('pauseFrom').value === moment(clickedButton.pauseFrom).format("YYYY/MM/DD hh:mm");
        var pauseToChange = document.getElementById('pauseTo').value === moment(clickedButton.pauseTo).format("YYYY/MM/DD hh:mm");
        var routineIntervalChange = document.getElementById('routineTypeField').value === clickedButton.repeatInterval;
        calChanged = !(beginChange && endChange && titleChange && pauseFromChange && pauseToChange && routineIntervalChange);
    }
    else {
        document.getElementById("httpMetoda").value = "post";
    }
    if (calChanged) {
        $('#warningModal').modal();
    } else {
        $('#myModal').modal('toggle');
    }
}

function deleteOptions() {
    document.getElementById("httpMetoda").value = "delete";
    document.getElementById("eventId").value = clickedButton.id;
    $('#warningModal').modal();
    eventFlag = "DELETE";
}

function sendPageDate() {
    var x = {
        "date": $('#calendar').fullCalendar('getView').start.format("l")
    };

    $.ajax({
        type: "POST",
        contentType: "application/json; charset=utf-8",
        url: "/calendar/page",
        data: JSON.stringify(x),
        error: function (xhr, ajaxOptions, thrownError) {
            alert(xhr.status);
            alert(thrownError);
        }
    });
}

function hasErrors() {
    var errorMessage1 = "Some fields are empty!";
    var errorMessage2 = "PauseTo field must be bigger or equal than PauseFrom field";
    var titleField = document.getElementById('titleField').value;
    var routineOptions = document.getElementById("toggle-event").checked;

    if (titleField === "") {
        document.getElementById("errorModalMessage").innerHTML = errorMessage1;
        $('#errorModal').modal();
        return true;
    }

    if (routineOptions === false) {
        return false;
    }

    var pauseToField = document.getElementById('pauseTo').value;
    var pauseFromField = document.getElementById('pauseFrom').value;

    if ((pauseFromField === "" && !(pauseToField === "")) || (!(pauseFromField === "") && pauseToField === "")) {
        document.getElementById("errorModalMessage").innerHTML = errorMessage1;
        $('#errorModal').modal();
        return true;
    }

    if (pauseToField < pauseFromField) {
        document.getElementById("errorModalMessage").innerHTML = errorMessage2;
        $('#errorModal').modal();
        return true;
    }
    return false;
}

function getRandomColor() {
    var letters = '0123456789ABCDEF';
    var color = '#';
    for (var i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
}

function deleteBtn(f){
    f.remove();
    var removeId = f.value;
    var url;

    for (var j = 0; j < sources.length; ++j){
        var url_id = sources[j].url.split("/")[2];
        console.log(url_id);
        if(removeId === url_id){
            url = sources[j].url;
            console.log("rm:" + sources[j].url);
            sources.splice(j,1);
            break;
        }
    }

    console.log(sources);
    $("#calendar").fullCalendar("removeEventSource",url);
   // $("#calendar").fullCalendar("refetchEventSources",sources);
}