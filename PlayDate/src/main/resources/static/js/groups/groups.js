function bs_input_file() {
	$(".input-file").before(
		function() {
			if ( ! $(this).prev().hasClass('input-ghost') ) {
				var element = $("<input type='file' class='input-ghost' style='visibility:hidden; height:0'>");
				element.attr("name",$(this).attr("name"));
				element.change(function(){
					element.next(element).find('input').val((element.val()).split('\\').pop());
				});
				$(this).find("button.btn-choose").click(function(){
					element.click();
				});
				$(this).find("button.btn-reset").click(function(){
					element.val(null);
					$(this).parents(".input-file").find('input').val('');
				});
				$(this).find('input').css("cursor","pointer");
				$(this).find('input').mousedown(function() {
					$(this).parents('.input-file').prev().click();
					return false;
				});
				return element;
			}
		}
	);
}
$(function() {
	bs_input_file();
});

function createPost(post) {
    console.debug(post);
    console.debug(post.id);
    console.debug(post.likedByCurrentUser);
    let postHTML = `
        <div id="post-${post.id}" class="card-post">
            <div class="row">
                <div class="col-xs-3 col-sm-2">
                    <a href="/user/${post.owner.username}" title="Profile">
                        <img src="/users/${post.owner.username}/image" alt="${post.owner.username}" class="img-circle img-user">
                    </a>
                </div>
                <div class="col-xs-9 col-sm-10 info-user">
                    <h3><a href="/user/${post.owner.username}" title="Profile">${post.owner.firstName} ${post.owner.lastName}</a></h3>
                    <p id="post-${post.id}-posted-before"><i class="prettydate">${post.datePosted}</i></p>
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
                        <input id="post-${post.id}-comment" onkeypress="addComment(event, ${post.id});" type="text" class="form-control" placeholder="Add a comment">
                    </div>
                </div>
            </div>
        </div>`;
    return postHTML;
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

function addToGroup(groupId, username){

  $.ajax({
      type: "POST",
      url: "/group/" + groupId + "/add/" + username,
      success: function () {
          location.reload(true);
        }
      });
}
