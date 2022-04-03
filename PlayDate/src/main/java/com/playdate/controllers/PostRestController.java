package com.playdate.controllers;

import com.playdate.enumerations.FilterType;
import com.playdate.enumerations.PostType;
import com.playdate.forms.PostForm;
import com.playdate.models.Comment;
import com.playdate.models.Group;
import com.playdate.models.Post;
import com.playdate.models.User;
import com.playdate.models.dtos.*;
import com.playdate.services.GroupService;
import com.playdate.services.PostService;
import com.playdate.services.UserService;
import com.playdate.utility.Utility;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static com.playdate.utility.Utility.NUMBER_OF_POSTS;

@RestController
public class PostRestController {

    private final UserService userService;
    private final GroupService groupService;
    private final PostService postService;

    @Autowired
    public PostRestController(UserService userService, PostService postService, GroupService groupService) {
        this.userService = userService;
        this.postService = postService;
        this.groupService = groupService;
    }

    @PostMapping(value = "/post/new")
    public void createPost(
            @RequestParam("content") String content,
            @RequestParam(value = "latitude", required = false) Double latitude,
            @RequestParam(value = "longitude", required = false) Double longitude,
            @RequestParam("postType") PostType postType,
            @RequestParam(value = "image", required = false) MultipartFile image,
            @RequestParam(value = "video", required = false) MultipartFile video) {
        if (content == null || content.isEmpty() || postType == null) {
            return;
        }
        User currentUser = userService.getCurrentUser();

        PostForm postForm = new PostForm();

        postForm.setContent(content);
        postForm.setLatitude(latitude);
        postForm.setLongitude(longitude);
        postForm.setPostType(postType);
        postForm.setImage(image);
        postForm.setVideo(video);

        Post post = postService.createPost(postForm, currentUser);
        currentUser.getPosts().add(post);
    }

    @PostMapping(value = "/post/diary/new")
    public void createDiaryPost(
            @RequestParam("content") String content,
            @RequestParam(value = "latitude", required = false) Double latitude,
            @RequestParam(value = "longitude", required = false) Double longitude,
            @RequestParam("postType") PostType postType,
            @RequestParam(value = "image", required = false) MultipartFile image,
            @RequestParam(value = "video", required = false) MultipartFile video) {
        if (content == null || content.isEmpty() || postType == null) {
            return;
        }
        User currentUser = userService.getCurrentUser();

        PostForm postForm = new PostForm();

        postForm.setContent(content);
        postForm.setLatitude(latitude);
        postForm.setLongitude(longitude);
        postForm.setPostType(postType);
        postForm.setImage(image);
        postForm.setVideo(video);

        Post post = postService.createDiaryPost(postForm, currentUser);
        currentUser.getPosts().add(post);
    }

    @PostMapping(value = "/post/wishlist/new")
    public void createWishlistPost(
            @RequestParam("content") String content,
            @RequestParam(value = "latitude", required = false) Double latitude,
            @RequestParam(value = "longitude", required = false) Double longitude,
            @RequestParam("postType") PostType postType,
            @RequestParam(value = "image", required = false) MultipartFile image,
            @RequestParam(value = "video", required = false) MultipartFile video) {
        if (content == null || content.isEmpty() || postType == null) {
            return;
        }
        User currentUser = userService.getCurrentUser();

        PostForm postForm = new PostForm();

        postForm.setContent(content);
        postForm.setLatitude(latitude);
        postForm.setLongitude(longitude);
        postForm.setPostType(postType);
        postForm.setImage(image);
        postForm.setVideo(video);

        Post post = postService.createWishlistPost(postForm, currentUser);
        currentUser.getPosts().add(post);
    }

    @PostMapping(value = "/group/{id}/post/new")
    public void createGroupPost(
            @PathVariable("id") Long id,
            @RequestParam("content") String content,
            @RequestParam(value = "latitude", required = false) Double latitude,
            @RequestParam(value = "longitude", required = false) Double longitude,
            @RequestParam("postType") PostType postType,
            @RequestParam(value = "image", required = false) MultipartFile image,
            @RequestParam(value = "video", required = false) MultipartFile video) {

        User currentUser = userService.getCurrentUser();
        Group group = groupService.findGroupById(id);

        if(!group.getMembers().contains(currentUser)){
            return;
        }

        if (content == null || content.isEmpty() || postType == null) {
            return;
        }
        PostForm postForm = new PostForm();

        postForm.setContent(content);
        postForm.setLatitude(latitude);
        postForm.setLongitude(longitude);
        postForm.setPostType(postType);
        postForm.setImage(image);
        postForm.setVideo(video);

        Post post = postService.createGroupPost(postForm, currentUser, group);
        currentUser.getPosts().add(post);
    }

    @PostMapping(value = "/post/{postId}/edit")
    public void editPost(
            @PathVariable("postId") Long postId,
            @RequestParam("content") String content,
            @RequestParam(value = "latitude", required = false) Double latitude,
            @RequestParam(value = "longitude", required = false) Double longitude,
            @RequestParam("postType") PostType postType,
            @RequestParam(value = "image", required = false) MultipartFile image,
            @RequestParam(value = "video", required = false) MultipartFile video) {
        if (content == null || content.isEmpty() || postType == null) {
            return;
        }
        User currentUser = userService.getCurrentUser();

        PostForm postForm = new PostForm();

        postForm.setContent(content);
        postForm.setLatitude(latitude);
        postForm.setLongitude(longitude);
        postForm.setPostType(postType);
        postForm.setImage(image);
        postForm.setVideo(video);

        postService.editPost(postId, postForm, currentUser);
    }

    @GetMapping("/post/{postId}/information")
    public EditPostInfo getEditPostInfo(@PathVariable("postId") Long postId) {
        return postService.getEditPostInfo(postId);
    }

    @GetMapping("/posts/{pageNumber}")
    public List<PostForFeed> fetchMorePostsForHomeFeed(@PathVariable("pageNumber") int pageNumber, @RequestParam(required = false) String filter) {
        User currentUser = userService.getCurrentUser();

        Set<User> users = userService.fetchUserFriends(currentUser.getUsername());
        users.add(currentUser);

        FilterType filterType = Utility.getFilterTypeForString(filter);

        if (filterType.equals(FilterType.ALL)) {
            return postService.fetchPostsForFeedFromUsers(users, pageNumber, NUMBER_OF_POSTS);
        } else {
            return postService.fetchPostsForFeedFromUsersByPostType(users, PostType.valueOf(filterType.toString()), pageNumber, NUMBER_OF_POSTS);
        }
    }

    //'/users/'+username+'/profile/posts/'+page
    @GetMapping("/users/{username}/profile/posts/{pageNumber}")
    public List<PostForFeed> fetchMorePostsForProfile(@PathVariable("username") String username,
                                                       @PathVariable("pageNumber") int pageNumber) {
        User user = userService.findByUsername(username);

        return postService.fetchPostsForProfileFromUser(user, pageNumber, NUMBER_OF_POSTS);
    }

    //'/users/' + username + '/diary/posts/' + page;
    @GetMapping("/users/{username}/diary/posts/{pageNumber}")
    public List<PostForDiary> fetchMorePostsForDiary(@PathVariable("username") String username,
                                                     @PathVariable("pageNumber") int pageNumber) {
        User user = userService.findByUsername(username);

        return postService.fetchPostsForDiaryFromUser(user, pageNumber, NUMBER_OF_POSTS);
    }

    //'/users/' + username + '/wishlist/posts/' + page;
    @GetMapping("/users/{username}/wishlist/posts/{pageNumber}")
    public List<PostForDiary> fetchMorePostsForWishlist(@PathVariable("username") String username,
                                                     @PathVariable("pageNumber") int pageNumber) {
        User user = userService.findByUsername(username);

        return postService.fetchPostsForWishlistFromUser(user, pageNumber, NUMBER_OF_POSTS);
    }

    @GetMapping("/user/{username}/activity")
    public List<PostForActivityMap> userActivityMap(@PathVariable("username") String username) {
        User user = userService.findByUsername(username);

        return postService.fetchPostsForActivityMapFromUser(user, Utility.MAX_POST_AGE);
    }

    @RequestMapping("/posts/{groupId}/{pageNumber}")
    public List<PostForFeed> fetchMorePostsForGroupFeed(@PathVariable("groupId") Long id, @PathVariable("pageNumber") int pageNumber) {
        Group group = this.groupService.findGroupById(id);
        return postService.fetchPostsForFeedFromGroup(group, pageNumber, NUMBER_OF_POSTS).stream().map(p -> postService.createPostForFeed(p.getId())).collect(Collectors.toList());
    }

    @PostMapping(value = "/post/{postId}/comment", consumes = "text/plain")
    public CommentDto addComment(@PathVariable("postId") Post post, @RequestBody String content) {
        User currentUser = userService.getCurrentUser();

        Comment comment = new Comment();

        comment.setOwner(currentUser);
        comment.setContent(content);

        post.addComment(comment);

        postService.flush();

        return new CommentDto(new UserDto(currentUser.getFirstName(), currentUser.getLastName(), currentUser.getUsername(), currentUser.getEmail(),currentUser.getId()), comment.getContent());
    }

    @PostMapping("/post/{postId}/like")
    public Integer like(@PathVariable(name = "postId") Post post) {
        User currentUser = userService.getCurrentUser();

        if (postService.isPostLikedBy(post.getId(), currentUser.getUsername())) {
            post.removeLikedBy(currentUser);
        } else {
            post.addLikedBy(currentUser);
        }

        postService.flush();

        return post.getLikedBy().size();
    }

    @PostMapping("/post/{postId}/resolve")
    public void resolve(@PathVariable(name = "postId") Post post) {
        if (postService.isPostResolved(post.getId())) {
            post.setClosed(false);
        } else {
            post.setClosed(true);
        }

        postService.flush();
    }

    @PostMapping("/post/{postId}/wish")
    public boolean wish(@PathVariable(name = "postId") Post post) {
        return postService.addRemoveWish(userService.getCurrentUser(),post);
    }

    @DeleteMapping("/post/{postId}/delete")
    public void deletePost(@PathVariable("postId") Long postId) {
        postService.deletePost(postId);
    }
}
