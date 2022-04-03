"use strict";

$(document).ready(function () {
    $('#search-box').on('keyup', _.debounce(function (e) {
        let query = $('#search-box').val();
        let result = $('#dropdown');

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
                    newHtml += `<a class="list-group-item list-group-item-action" href="/user/${users[i].username}">
                                            <div class="row mb-1 mt-1">
                                                <div class="col-auto">
                                                    <img class="user-img" src="/users/${users[i].username}/image">
                                                </div>
                                                <div class="col-auto">
                                                    <div>${users[i].firstName} ${users[i].lastName}</div>
                                                </div>
                                            </div>
                                        </a>`;
                }
                for (let i = 0; i < groups.length; i++) {
                    newHtml += `<a class="list-group-item list-group-item-action" href="/group/${groups[i].id}">
                                            <div class="row mb-1 mt-1">
                                                <div class="col-auto">
                                                    <img class="user-img" src="/groups/${groups[i].id}/banner">
                                                </div>
                                                <div class="col-auto">
                                                    <div>${groups[i].groupName}</div>
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

function filterChanged(source) {
    let filterType = source.value;

    window.location.replace("/wall?filter="+filterType);
}

function appendError(message, appendAfterId) {
    let reference = document.getElementById(appendAfterId);

    let errorDiv = document.createElement('div');
    errorDiv.innerHTML = message;
    errorDiv.className = 'to-remove alert alert-danger';
    errorDiv.role = 'alert';

    reference.parentNode.insertBefore(errorDiv, reference.nextSibling);
}

function appendSuccess(message, appendAfterId) {
    let reference = document.getElementById(appendAfterId);

    let errorDiv = document.createElement('div');
    errorDiv.innerHTML = message;
    errorDiv.className = 'to-remove alert alert-success';
    errorDiv.role = 'alert';

    reference.parentNode.insertBefore(errorDiv, reference.nextSibling);
}

function fillModalWithInformation(modalId, postInfo) {
    $('#content').val(postInfo.content);

    let postType = postInfo.postType;
    $("select ").val(postType);

    if (postInfo.image) {
        $(`#imgIcon`).hide();
        $(`#imgImg`).attr('src', '/post/' + postInfo.id + '/image');
        $(`#imgImg`).css('display', 'block');
        $(`#imgBtn`).css('border', 0);
    }
    if (postInfo.video) {
        $('#iconVid').hide();
        $('#btnVid').css('border', 0);
        let $source = $('#video_here');
        $source[0].src = '/post/' + postInfo.id + '/video';
        $source.parent()[0].load();
        $source.parent().css("display", "");
    }

    if (postInfo.latitude && postInfo.longitude) {
        let latitude = postInfo.latitude;
        let longitude = postInfo.longitude;

        let viewLocation = {latitude, longitude, zoomLevel: 14};
        let pinLocation = {latitude, longitude};

        showMap(viewLocation, pinLocation, pinLocation);
    } else {
        showMap();
    }
}

function showModal(modalId) {
    $(modalId).modal('show');
}

function reloadPage(modalId) {
    $(modalId).on('hidden.bs.modal', function () {
        location.reload();

        $(modalId).on('hidden.bs.modal', function () {
            // do nothing
        });
    });
}

function resetModalOnClose(modalId, clearInput) {
    $(modalId).on('hidden.bs.modal', function () {
        resetModal(modalId, clearInput);

        $(modalId).on('hidden.bs.modal', function () {
            // do nothing
        });
    });
}

let postModalCopy = $('#postModal').clone();

function absolutePostModalReset() {
    $('#postModal').on('hidden.bs.modal', function () {
        $('#postModal').replaceWith(postModalCopy.clone());

        $('#postModal').on('hidden.bs.modal', function () {
            // do nothing
        });
    });
}

function resetModal(modalId, clearInput) {
    if (clearInput) {
        $(modalId).find('input').val('');
        $(modalId).find('textarea').val('');
    }
    $(modalId).find('.to-remove').remove();
}


function hideAndResetModalOnClose(modalId, clearInput) {
    resetModalOnClose(modalId, clearInput);
    $(modalId).modal('hide');
}

function isEmailValid(email) {
    let emailPattern = /^(([^<>()[\]\\.,;:\s@\"]+(\.[^<>()[\]\\.,;:\s@\"]+)*)|(\".+\"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
    return emailPattern.test(email);
}

function isPhoneValid(phone) {
    let phonePattern = /^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$/im;
    return phonePattern.test(phone);
}