"use strict";

let map;
function showMap(viewLocation, pinLocation, circleLocation) {
    if (map) {
        map.off();
        map.remove();
    }
    let latitude = 40.91;
    let longitude = -96.63;
    let zoomLevel = 4;

    if (viewLocation) {
        initializeMap(viewLocation, pinLocation, circleLocation);
    } else {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(function (position) {
                latitude = position.coords.latitude;
                longitude = position.coords.longitude;

                viewLocation = {latitude, longitude, zoomLevel: 14};

                initializeMap(viewLocation, pinLocation, viewLocation);
            });
        } else {
            initializeMap({latitude, longitude, zoomLevel: 4}, null, null);
        }
    }
}

function initializeMap(viewLocation, pinLocation, circleLocation) {
    map = L.map('map');

    map.on('load', function () {
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
        setInterval(function () {
            map.invalidateSize();
        }, 200);
    });
    map.setView([viewLocation.latitude, viewLocation.longitude], viewLocation.zoomLevel);

    // L.esri.basemapLayer('Imagery').addTo(map);
    // L.esri.basemapLayer('ImageryLabels').addTo(map);


    let searchControl = L.esri.Geocoding.geosearch().addTo(map);

    let results = L.layerGroup().addTo(map);

    if (pinLocation) {
        results.addLayer(L.marker({lat: pinLocation.latitude, lng: pinLocation.longitude}));
    }

    if (circleLocation) {
        let circle = L.circle([circleLocation.latitude, circleLocation.longitude], {
            color: 'red',
            fillColor: '#ff337d',
            fillOpacity: 0.3,
            radius: 500,
            opacity: 0.3
        }).addTo(map);
    }

    let latitude = $('#latitude');
    let longitude = $('#longitude');

    searchControl.on('results', function (data) {
        results.clearLayers();
        for (let i = data.results.length - 1; i >= 0; i--) {
            let latlng = data.results[i].latlng;

            results.addLayer(L.marker(latlng));
            latitude.val(latlng.lat);
            longitude.val(latlng.lng);

            console.debug('Latitude: ' + latlng.lat);
            console.debug('Longitude: ' + latlng.lng);

        }
    });

    map.on('click', function (e) {
        results.clearLayers();

        let latlng = e.latlng;

        results.addLayer(L.marker(latlng));

        latitude.val(latlng.lat);
        longitude.val(latlng.lng);

        console.debug('Latitude: ' + latlng.lat);
        console.debug('Longitude: ' + latlng.lng);

    });
}

function showEditPost(postId) {
    absolutePostModalReset();
    resetModal('#postModal', true);        //remove all input values and errors
    $("#publishBtn").attr("onclick", `editPost(${postId})`);
    $.ajax({
        url: '/post/' + postId + '/information',
        type: "GET",
        success: function (postInfo) {
            console.debug(postInfo);
            fillModalWithInformation('#postModal', postInfo);
            showModal('#postModal');
        }, error: function () {
            console.debug('err');
        }
    });
}

function editPost(postId) {
    if (!isPostFormDataValid()) return;

    let data = extractPostFormData();

    $.ajax({
        url: '/post/' + postId + '/edit',
        data: data,
        enctype: 'multipart/form-data',
        contentType: false,
        processData: false,
        cache: false,
        type: 'POST',
        success: function () {
            appendSuccess('Post edited!', 'create-post-message-area');
            $('#publishBtn').prop('disabled', true).css('opacity', 0.5).css('background-color', '#999').css('border-color', '#999').css('outline', 'none').css('box-shadow', 'none');
            $('#cancelBtn').hide();
            reloadPage('#postModal');
        }, error: function () {
            appendError('An error occurred while editing the post!', 'create-post-message-area');
        }
    });
}

function showCreateNewWishlistPost() {
    absolutePostModalReset();
    resetModal('#postModal', true);        //remove all input values and errors
    $("#publishBtn").attr("onclick", "createNewWishListPost()");
    resetModalOnClose('#postModal', true);
    showModal('#postModal');
    showMap();
}

function createNewWishListPost() {
    if (!isPostFormDataValid()) return;

    let data = extractPostFormData();

    $.ajax({
        url: '/post/wishlist/new',
        data: data,
        enctype: 'multipart/form-data',
        contentType: false,
        processData: false,
        cache: false,
        type: 'POST',
        success: function () {
            appendSuccess('Wishlist post created!', 'create-post-message-area');
            $('#publishBtn').prop('disabled', true).css('opacity', 0.5).css('background-color', '#999').css('border-color', '#999').css('outline', 'none').css('box-shadow', 'none');
            $('#cancelBtn').hide();
            reloadPage('#postModal');
        }, error: function () {
            appendError('An error occurred while creating the post!', 'create-post-message-area');
        }
    });
}

function showCreateNewDiaryPost() {
    absolutePostModalReset();
    resetModal('#postModal', true);        //remove all input values and errors
    $("#publishBtn").attr("onclick", "createNewDiaryPost()");
    resetModalOnClose('#postModal', true);
    showModal('#postModal');
    showMap();
}

function createNewDiaryPost() {
    if (!isPostFormDataValid()) return;

    let data = extractPostFormData();

    $.ajax({
        url: '/post/diary/new',
        data: data,
        enctype: 'multipart/form-data',
        contentType: false,
        processData: false,
        cache: false,
        type: 'POST',
        success: function () {
            appendSuccess('Diary post created!', 'create-post-message-area');
            $('#publishBtn').prop('disabled', true).css('opacity', 0.5).css('background-color', '#999').css('border-color', '#999').css('outline', 'none').css('box-shadow', 'none');
            $('#cancelBtn').hide();
            reloadPage('#postModal');
        }, error: function () {
            appendError('An error occurred while creating the post!', 'create-post-message-area');
        }
    });
}

function showCreateNewPost() {
    absolutePostModalReset();
    resetModal('#postModal', true);        //remove all input values and errors
    $("#publishBtn").attr("onclick", "createNewPost()");
    resetModalOnClose('#postModal', true);
    showModal('#postModal');
    showMap();
}

function createNewPost() {
    if (!isPostFormDataValid()) return;

    let data = extractPostFormData();

    $.ajax({
        url: '/post/new',
        data: data,
        enctype: 'multipart/form-data',
        contentType: false,
        processData: false,
        cache: false,
        type: 'POST',
        success: function () {
            appendSuccess('Post created!', 'create-post-message-area');
            $('#publishBtn').prop('disabled', true).css('opacity', 0.5).css('background-color', '#999').css('border-color', '#999').css('outline', 'none').css('box-shadow', 'none');
            $('#cancelBtn').hide();
            reloadPage('#postModal');
        }, error: function () {
            appendError('An error occurred while creating the post!', 'create-post-message-area');
        }
    });
}

function extractPostFormData() {
    let content = $('#content').val();
    let latitude = $('#latitude').val();
    let longitude = $('#longitude').val();
    let postType = $('#postType').val();

    let data = new FormData();

    data.append('image', $('#photos input[type=file]')[0].files[0]);
    data.append('video', $('#videos input[type=file]')[0].files[0]);

    data.append('content', content);

    if (longitude) data.append('longitude', longitude);
    if (latitude) data.append('latitude', latitude);

    data.append('postType', postType);

    console.debug(data);

    return data;
}

function isPostFormDataValid() {
    resetModal('#postModal', false);

    if (!$('#content').val()) {
        appendError('Posts text content must not be empty', 'create-post-message-area');
        return false;
    }

    return true;
}

// Loads video on post modal
$(document).on("change", ".file_multi_video", function (evt) {
    $('#iconVid').hide();
    $('#btnVid').css('border', 0);
    let $source = $('#video_here');
    $source[0].src = URL.createObjectURL(this.files[0]);
    $source.parent()[0].load();
    $source.parent().css("display", "");
});

// Loads image on post modal
function readURL(input) {
    if (input.files && input.files[0]) {
        let reader = new FileReader();

        if (input.files[0].size > 10485760) {
            appendError('Image file too big', 'photos')
            return;
        }

        let image = new Image();

        reader.onload = function (e) {
            image.onload = function () {
                $(`#imgIcon`).hide();
                $(`#imgImg`).attr('src', e.target.result);
                $(`#imgImg`).css('display', 'block');
                $(`#imgBtn`).css('border', 0);
            };

            image.onerror = function () {
                appendError('Invalid file', 'photos')
                return;
            };

            image.src = e.target.result;
        };

        reader.readAsDataURL(input.files[0]);
    }
}


function addComment(event, postId) {
    if (event.keyCode !== 13) return;    // press enter to submit

    let content = $('#post-' + postId + '-comment').val();

    if (!content) return;


    $.ajax({
        url: '/post/' + postId + '/comment',
        type: "POST",
        data: content,
        contentType: "text/plain",
        dataType: "json",
        success: function (comment) {
            let owner = comment.owner;
            let content = comment.content;

            let commentHTML = `
            <li><b>${owner.firstName} ${owner.lastName}</b> ${content}</li>
            `;

            $('#post-' + postId + '-comments ul').append(commentHTML);
            $('#post-' + postId + '-comment').val('');
        }
    });
}

function addToWishlist(postId, isWishlistPage) {
    let wish = $('#post-' + postId + '-wish');

    if (wish.hasClass('hoverable')) {
        wish.removeClass('hoverable');
    } else {
        wish.addClass('hoverable');
    }
    $.ajax({
        type: "POST",
        url: '/post/' + postId + '/wish',
        success: function (wasRemoved) {
            if (wasRemoved && isWishlistPage==='true') {
                $('#post-' + postId).remove();
            }
        }
    });
}

function likePost(event, postId) {
    let element = $('#post-' + postId + '-like');

    if (element.hasClass('not-liked')) {
        element.html('<i class="fa fa-heart" aria-hidden="true"></i> You liked this');
        element.children('.fa-heart').addClass('animate-like');
        element.removeClass('not-liked').addClass('liked');
    } else {
        element.html('<i class="fa fa-heart" aria-hidden="true"></i> Like');
        element.removeClass('liked').addClass('not-liked');
    }

    $.ajax({
        type: "POST",
        url: '/post/' + postId + '/like'
    });
}

function resolvePost(event, postId) {
    let element = $('#post-' + postId + '-resolve');
    let elementIcon = $('#post-' + postId + '-resolve-icon');
    let comment = $('#post-' + postId + '-comment');

    if (element.hasClass('not-resolved')) {
        // element.text('Resolved');
        element.removeClass('badge-warning not-resolved').addClass('badge-success resolved');
        elementIcon.removeClass('hoverable');
        comment.attr('disabled', true);
        comment.attr('readonly', true);
        comment.attr('placeholder', 'Post already resolved');
    } else {
        // element.text('Not Resolved');
        element.removeClass('badge-success resolved').addClass('badge-warning not-resolved');
        elementIcon.addClass('hoverable');
        comment.attr('disabled', false);
        comment.attr('readonly', false);
        comment.attr('placeholder', 'Add a comment');
    }

    $.ajax({
        type: "POST",
        url: '/post/' + postId + '/resolve'
    });
}

function deletePost(event, postId) {
    bootbox.confirm({
        // title: "Delete Post",
        message: "Are you sure you want to delete this post?",
        size: "large",
        buttons: {
            cancel: {
                label: '<i class="fa fa-times"></i> Cancel',
                className: 'btn btn-primary'
            },
            confirm: {
                label: '<i class="fa fa-check"></i> Confirm',
                className: 'btn btn-primary'
            }
        },
        callback: function (result) {
            if (result) {
                $.ajax({
                    type: "DELETE",
                    url: '/post/' + postId + '/delete',
                    success: function () {
                        $('#post-' + postId).remove();
                    }
                });
            }
        }
    });
}