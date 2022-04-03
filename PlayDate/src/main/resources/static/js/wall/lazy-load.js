"use strict";

function scrollOnWall(apiURL, scrollInfo) {

    $.ajax({
        type: "GET",
        url: apiURL,
        success: function (posts) {
            setTimeout(
                function () {
                    let postsHTML = '';
                    $.each(posts, function (i, item) {
                        postsHTML += createPost(item, scrollInfo.isAdmin, scrollInfo.username, scrollInfo.currentUserUsername, scrollInfo.pageName, scrollInfo.myPage);
                    })

                    $("#posts-container").append(postsHTML);
                    $(".prettydate").prettydate();
                    $("#loading").hide();

                    scrollInfo.scrolled = false;
                }, 1e3)
        }
    });
}

function createPost(post, isAdmin, username, currentUserUsername, pageName, myPage) {
    console.debug(post);
    console.debug(post.id);
    console.debug(post.likedByCurrentUser);
    let author = post.owner;
    let postHTML = `
        <div id="post-${post.id}" class="card-post ${post.postType}">
            <div class="row">
                <div class="col-xs-3 col-sm-2">
                    <a href="/user/${author.username}" title="Profile">
                        <img src="/users/${author.username}/image" alt="${author.username}" class="img-circle img-user">
                    </a>
                </div>
                <div class="col-xs-9 col-sm-10 info-user">
                 <div class="row">
                <div class="col-5 p-l-0 mr-auto">
                    <h3><a href="/user/${author.username}" title="Profile">${author.firstName} ${author.lastName}</a>` +
        appendSpanIfNotTextPost(post) +
        appendSpanIfEdited(post) +
        `</h3>
                </div>
                <div class="col-2 offset-10 ml-auto">
                    <div class="row justify-content-end p-r-20">` +
        appendIfPageDiaryOrWishlistAndMyPage(post, currentUserUsername,pageName, myPage)+
        appendSpanIfNotTextPostAndCurrentUserNotAuthor(post, currentUserUsername) +
        appendSpanIfCurrentUserAuthor(post, currentUserUsername) +
        appendSpanIfCurrentUserAuthorOrAdmin(post, currentUserUsername, isAdmin)

        + `</div>
                   </div>
                </div>    
                <div class="row">
                    <p id="post-${post.id}-posted-before"><i class="prettydate">${post.datePosted}</i></p>
                </div>
                </div>
            </div>
            <div class="row">
                <div class="col-sm-8 col-sm-offset-2 data-post">
                    <p id="post-${post.id}-content">${post.content}</p>` +
        appendImage(post) +
        appendVideo(post)
        + `<div class="reaction">
                        <div class="like-content">
                            <button id="post-${post.id}-like" class="btn-secondary like-review ${post.likedByCurrentUser ? 'liked' : 'not-liked'}" onclick="likePost(event,${post.id});">
                                <i class="fa fa-heart" aria-hidden="true"></i> ${post.likedByCurrentUser ? 'You liked this' : 'Like'}
                            </button>
                        </div>
                    </div>
                    <div id="post-${post.id}-comments" class="comments">
                        <div class="more-comments">View more comments</div>
                        <ul>
                        ` +
        insertComments(post)
        + `
                        </ul>
                        <input id="post-${post.id}-comment" 
                        onkeypress="addComment(event, ${post.id});" 
                        type="text" 
                         ${post.resolved ? 'disabled' : ''} ${post.resolved ? 'readonly' : ''} 
                        class="form-control" 
                        placeholder="${post.resolved ? 'Post already resolved' : 'Add a comment'}">
                    </div>
                </div>
            </div>
        </div>`;
    console.debug(postHTML);
    return postHTML;
}

function appendIfPageDiaryOrWishlistAndMyPage(post, currentUserUsername, pageName, myPage) {
    let spanString = ``;
    if (pageName === 'DIARY' || (pageName === 'WISHLIST' && myPage)) {
        let classAppend = 'hoverable';
        if (post.isInWishlist) {
            classAppend = '';
        }
        spanString = `
        <span id="post-${post.id}-wish"
                              class="m-r-10 fa fa-star clickable ${classAppend}"
                              onclick="addToWishlist(${post.id},${pageName==='WISHLIST'});" title="Wish"></span>
    `;
    }

    return spanString
}

function appendSpanIfCurrentUserAuthorOrAdmin(post, currentUserUsername, isAdmin) {
    let spanString = ``;
    if (currentUserUsername === post.owner.username || isAdmin) {
        spanString = `
        <span class="m-r-10 fa fa-trash hoverable"
              onclick="deletePost(event,${post.id});" title="Remove"></span>
    `;
    }

    return spanString
}

function appendSpanIfCurrentUserAuthor(post, currentUserUsername) {
    let spanString = ``;
    if (currentUserUsername === post.owner.username) {
        spanString = `
         <span class="m-r-10 fa fa-cog hoverable"
               onclick="showEditPost(${post.id});" title="Edit"></span>
    `;
    }

    return spanString
}

function appendSpanIfNotTextPostAndCurrentUserNotAuthor(post, currentUserUsername) {
    let spanString = ``;
    if (post.postType !== 'TEXT' && currentUserUsername === post.owner.username) {
        let classAppend = 'hoverable';
        if (post.resolved) {
            classAppend = '';
        }

        spanString = `
        <span class="m-r-10 fa fa-check clickable ${classAppend}" id="post-${post.id}-resolve-icon"
            onclick="resolvePost(event,${post.id});" title="Resolve"></span>
    `;
    }

    return spanString
}

function appendSpanIfNotTextPost(post) {
    let spanString = ``;

    if (post.postType !== 'TEXT') {
        let classAppend = 'badge-success resolved';
        if (!post.resolved) {
            classAppend = 'badge-warning not-resolved';
        }
        spanString += `
        <span id="post-${post.id}-resolve" class="badge m-l-10 ${classAppend}">${post.postType}</span>
        `;
    }

    return spanString;
}

function appendSpanIfEdited(post) {
    let spanString = ``;

    if (post.edited) {
        spanString += `
            <span class="badge m-l-10 badge-success">Edited</span>
        `;
    }

    return spanString;
}

function appendImage(post) {
    let imageString = ``;

    if (post.image) {
        imageString += `
            <img src="/post/${post.id}/image" class="img-post">
        `;
    }

    return imageString;
}

function appendVideo(post) {
    let videoString = ``;

    if (post.video) {
        videoString += `
           <video width="100%" height="auto" controls>
                    <source src="/post/${post.id}/video">
                    Your browser doesn't support the video tag.
                </video>
        `;
    }

    return videoString;
}


function insertComments(post) {
    let commentsHTML = '';
    let comments = post.comments;

    for (let i = 0; i < comments.length; i++) {
        let comment = comments[i];

        commentsHTML += `
             <li th:inline="text">
             <b>${comment.owner.firstName} ${comment.owner.lastName}</b> ${comment.content}</li>
        `
    }

    return commentsHTML;
}