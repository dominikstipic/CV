package com.playdate.services;

import com.playdate.enumerations.PostType;
import com.playdate.forms.PostForm;
import com.playdate.models.*;
import com.playdate.models.dtos.*;
import com.playdate.repositories.DiaryRepository;
import com.playdate.repositories.PostRepository;
import com.playdate.repositories.UserRepository;
import com.playdate.repositories.WishlistRepository;
import com.playdate.stores.ImageStore;
import com.playdate.stores.VideoStore;
import com.playdate.utility.Utility;
import lombok.SneakyThrows;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Calendar;
import java.util.*;
import java.util.stream.Collectors;

@Service
@Transactional
public class PostService {

    private final PostRepository postRepository;

//    @Value("${com.playdate.picture.storage}")
//    private String pictureStorageFolder;
//
//    @Value("${com.playdate.video.storage}")
//    private String videoStorageFolder;

    private final ImageStore imageStore;

    private final VideoStore videoStore;

    private final DiaryRepository diaryRepository;

    @Autowired
    private WishlistRepository wishlistRepository;

    @Autowired
    private UserRepository userRepository;


    @Autowired
    public PostService(PostRepository postRepository, ImageStore imageStore, VideoStore videoStore, DiaryRepository diaryRepository) {
        this.postRepository = postRepository;
        this.imageStore = imageStore;
        this.videoStore = videoStore;
        this.diaryRepository = diaryRepository;
    }

    public List<Post> findAllPostsOrderByDatePostedDesc() {
        return postRepository.findAllByOrderByDatePostedDesc();
    }

    public void flush() {
        postRepository.flush();
    }

    @Transactional
    public PostForFeed createPostForFeed(Long postId) {
        Post post = postRepository.findById(postId).get();
        User owner = post.getOwner();

        org.springframework.security.core.userdetails.User principal = (org.springframework.security.core.userdetails.User) SecurityContextHolder.getContext().getAuthentication().getPrincipal();

        return new PostForFeed(
                post.getId(),
                new UserDto(owner.getFirstName(), owner.getLastName(), owner.getUsername(), owner.getEmail(), owner.getId()),
                post.getDatePosted(),
                post.getContent(),
                post.getType(),
                post.getComments().stream().map(c -> new CommentDto(new UserDto(c.getOwner().getFirstName(), c.getOwner().getLastName(), c.getOwner().getUsername(), c.getOwner().getEmail(), c.getOwner().getId()), c.getContent())).collect(Collectors.toSet()),
                post.getLikedBy().stream().map(u -> u.getUsername()).anyMatch(u -> Objects.equals(u, principal.getUsername())),
                post.getClosed(),
                post.getPostImage() != null,
                post.getPostVideo() != null,
                post.getDateModified() != null);
    }

    @Transactional
    public PostForDiary createPostForDiary(Long postId) {
        PostForFeed postForFeed = createPostForFeed(postId);
        Post post = postRepository.findById(postId).get();

        Optional<User> userOptional = userRepository.findByUsername(((org.springframework.security.core.userdetails.User) SecurityContextHolder.getContext().getAuthentication().getPrincipal()).getUsername());
        User currentUser = userOptional.get();

        return new PostForDiary(
                postForFeed.getId(),
                postForFeed.getOwner(),
                postForFeed.getDatePosted(),
                postForFeed.getContent(),
                postForFeed.getPostType(),
                postForFeed.getComments(),
                postForFeed.isLikedByCurrentUser(),
                postForFeed.isResolved(),
                postForFeed.isImage(),
                postForFeed.isVideo(),
                postForFeed.isEdited(),
                post.getWishlists().contains(currentUser.getWishList())
        );
    }

    @Transactional
    public PostForActivityMap createPostForActivityMap(Long postId) {
        Post post = postRepository.findById(postId).get();
        User owner = post.getOwner();

        Location location = post.getLocation();
        return new PostForActivityMap(
                postId,
                new UserDto(owner.getFirstName(), owner.getLastName(), owner.getUsername(), owner.getEmail(), owner.getId()),
                post.getContent(),
                location != null ? location.getLatitude() : null,
                location != null ? location.getLongitude() : null,
                post.getDatePosted()
        );
    }

    @Transactional
    public boolean isPostLikedBy(Long postId, String username) {
        Optional<Post> post = postRepository.findById(postId);
        return post.map(post1 -> post1.getLikedBy().stream().map(User::getUsername).anyMatch(u -> Objects.equals(u, username))).orElse(false);
    }

    @Transactional
    public List<PostForFeed> fetchPostsForProfileFromUser(User user, int pageNumber, int numberOfPosts) {
        Set<User> users = new LinkedHashSet<>();
        users.add(user);

        return fetchPostsForFeedFromUsers(users, pageNumber, numberOfPosts);
    }

    @Transactional
    public List<PostForFeed> fetchPostsForFeedFromUsers(Set<User> users, int pageNumber, int numberOfPosts) {
        return fetchPostsFromUsers(users, pageNumber, numberOfPosts).stream().map(p -> createPostForFeed(p.getId())).collect(Collectors.toList());
    }

    @Transactional
    public List<PostForFeed> fetchPostsForFeedFromUsersByPostType(Set<User> users, PostType postType, int pageNumber, int numberOfPosts) {
        return fetchPostsFromUsersByPostType(users, postType, pageNumber, numberOfPosts).stream().map(p -> createPostForFeed(p.getId())).collect(Collectors.toList());
    }


    @Transactional
    public List<Post> fetchPostsFromUsers(Set<User> users, int pageNumber, int numberOfPosts) {
        return postRepository
                .findByOwnerIsIn(
                        users,
                        PageRequest.of(
                                pageNumber,
                                numberOfPosts,
                                Sort.by(
                                        Sort.Direction.DESC,
                                        "datePosted"))).getContent();
    }

    @Transactional
    public List<PostForFeed> fetchPostsForFeedFromGroup(Group group, int pageNumber, int numberOfPosts) {
        return fetchPostsFromGroup(group, pageNumber, numberOfPosts).stream().map(p -> createPostForFeed(p.getId())).collect(Collectors.toList());
    }

    @Transactional
    public List<Post> fetchPostsFromGroup(Group group, int pageNumber, int numberOfPosts) {
        return postRepository
                .findByGroupEquals(
                        group,
                        PageRequest.of(
                                pageNumber,
                                numberOfPosts,
                                Sort.by(
                                        Sort.Direction.DESC,
                                        "datePosted"))).getContent();
    }

    @Transactional
    public List<Post> fetchPostsFromUsersByPostType(Set<User> users, PostType postType, int pageNumber, int numberOfPosts) {
        return postRepository
                .findByOwnerIsInAndType(
                        users,
                        postType,
                        PageRequest.of(
                                pageNumber,
                                numberOfPosts,
                                Sort.by(
                                        Sort.Direction.DESC,
                                        "datePosted"))).getContent();
    }

    @Transactional
    public List<PostForDiary> fetchPostsForDiaryFromUser(User user, int pageNumber, int numberOfPosts) {
        Optional<User> userOptional = userRepository.findById(user.getId());
        Diary diary = userOptional.get().getDiary();

        return fetchPostsFromDiary(diary, pageNumber, numberOfPosts).stream().map(p -> createPostForDiary(p.getId())).collect(Collectors.toList());
    }

    @Transactional
    public List<Post> fetchPostsFromDiary(Diary diary, int pageNumber, int numberOfPosts) {
        return postRepository
                .findByDiaryIsIn(
                        diary,
                        PageRequest.of(
                                pageNumber,
                                numberOfPosts,
                                Sort.by(
                                        Sort.Direction.DESC,
                                        "datePosted"))).getContent();
    }

    @Transactional
    public List<PostForDiary> fetchPostsForWishlistFromUser(User user, int pageNumber, int numberOfPosts) {
        Optional<User> userOptional = userRepository.findById(user.getId());
        Wishlist wishlist = userOptional.get().getWishList();

        return fetchPostsFromWishlist(wishlist, pageNumber, numberOfPosts).stream().map(p -> createPostForDiary(p.getId())).collect(Collectors.toList());
    }

    @Transactional
    public List<Post> fetchPostsFromWishlist(Wishlist wishlist, int pageNumber, int numberOfPosts) {
        return postRepository
                .findByWishlistsIsIn(
                        wishlist,
                        PageRequest.of(
                                pageNumber,
                                numberOfPosts,
                                Sort.by(
                                        Sort.Direction.DESC,
                                        "datePosted"))).getContent();
    }

    @Transactional
    public List<PostForActivityMap> fetchPostsForActivityMapFromUser(User user, int maxDaysOld) {
        Optional<User> userOptional = userRepository.findById(user.getId());
        ActivityMap activityMap = userOptional.get().getActivityMap();

        return activityMap.getPosts().stream().map(p -> createPostForActivityMap(p.getId())).filter(p -> Utility.calcualteDays(p.getDatePosted()) <= Utility.MAX_POST_AGE).collect(Collectors.toList());
    }

    @Transactional
    @SneakyThrows(IOException.class)
    public Post createPost(PostForm postForm, User postOwner) {
        Post post = new Post();
        postOwner = userRepository.findById(postOwner.getId()).get();

        post.setContent(postForm.getContent());
        post.setType(postForm.getPostType());

        Double longitude = postForm.getLongitude();
        Double latitude = postForm.getLatitude();
        addPostLocation(post, longitude, latitude);
        if (post.getLocation() != null) {
            postOwner.getActivityMap().addPost(post);
        }

        MultipartFile image = postForm.getImage();
        addPostImage(post, image);

        MultipartFile video = postForm.getVideo();
        addPostVideo(post, video);

        post.setOwner(postOwner);

        return postRepository.saveAndFlush(post);
    }

    @Transactional
    public Post createDiaryPost(PostForm postForm, User author) {
        Post post = createPost(postForm, author);

        post.setDiary(author.getDiary());

        return postRepository.saveAndFlush(post);
    }

    @Transactional
    public Post createWishlistPost(PostForm postForm, User author) {
        Post post = createPost(postForm, author);

        author.getWishList().addPost(post);

        return postRepository.saveAndFlush(post);
    }

    @Transactional
    @SneakyThrows(IOException.class)
    public Post createGroupPost(PostForm postForm, User postOwner, Group group) {
        Post post = new Post();

        post.setContent(postForm.getContent());
        post.setType(postForm.getPostType());

        Double longitude = postForm.getLongitude();
        Double latitude = postForm.getLatitude();
        addPostLocation(post, longitude, latitude);

        MultipartFile image = postForm.getImage();
        addPostImage(post, image);

        MultipartFile video = postForm.getVideo();
        addPostVideo(post, video);

        post.setOwner(postOwner);
        group.addPost(post);

        return postRepository.saveAndFlush(post);
    }


    @Transactional
    public void deletePost(Long postId) {
        Optional<Post> post = postRepository.findById(postId);

        post.ifPresent(p -> postRepository.delete(p));
    }

    @Transactional
    public boolean isPostResolved(Long postId) {
        Optional<Post> post = postRepository.findById(postId);

        if (post.isPresent()) {
            return post.get().getClosed();
        }

        return false;
    }

    public EditPostInfo getEditPostInfo(Long postId) {
        Optional<Post> post = postRepository.findById(postId);

        if (post.isPresent()) {
            Post p = post.get();

            Location l = p.getLocation();

            return new EditPostInfo(
                    p.getId(),
                    p.getContent(),
                    l != null ? l.getLongitude() : null,
                    l != null ? l.getLatitude() : null,
                    p.getType(),
                    p.getPostImage() != null,
                    p.getPostVideo() != null
            );
        }

        return null;
    }

    @Transactional
    @SneakyThrows(IOException.class)
    public void editPost(Long postId, PostForm postForm, User currentUser) {
        Optional<Post> postOptional = postRepository.findById(postId);
        Optional<User> author = userRepository.findById(currentUser.getId());

        if (!postOptional.isPresent()) return;

        Post post = postOptional.get();

        post.setContent(postForm.getContent());
        post.setType(postForm.getPostType());
        if (post.getType().equals(PostType.TEXT)) post.setClosed(false);

        Double longitude = postForm.getLongitude();
        Double latitude = postForm.getLatitude();
        addPostLocation(post, longitude, latitude);

        MultipartFile image = postForm.getImage();
        addPostImage(post, image);

        MultipartFile video = postForm.getVideo();
        addPostVideo(post, video);

        post.setDateModified(Calendar.getInstance().getTime());
    }

    private void addPostLocation(Post post, Double longitude, Double latitude) {
        if (longitude != null && latitude != null) {
            Location location = new Location();

            location.setLongitude(longitude);
            location.setLatitude(latitude);

            if (post.getLocation() == null) {
                location.setPost(post);
                post.setLocation(location);
            } else {
                post.getLocation().setLatitude(latitude);
                post.getLocation().setLongitude(longitude);
            }

        }
    }

    private void addPostVideo(Post post, MultipartFile video) throws IOException {
        if (video != null) {
            Video vid = new Video();
            vid.setMimeType(video.getContentType());

            vid.setPost(post);
            post.setPostVideo(vid);

            videoStore.setContent(vid, video.getInputStream());
        }
    }

    private void addPostImage(Post post, MultipartFile image) throws IOException {
        if (image != null) {
            Image img = new Image();
            img.setMimeType(image.getContentType());

            img.setPost(post);
            post.setPostImage(img);

            imageStore.setContent(img, image.getInputStream());
        }
    }

    @Transactional
    public boolean addRemoveWish(User currentUser, Post post) {
        Optional<User> userOptional = userRepository.findById(currentUser.getId());
        Optional<Post> postOptional = postRepository.findById(post.getId());
        currentUser = userOptional.get();
        post = postOptional.get();

        if (post.getWishlists().contains(currentUser.getWishList())) {
            currentUser.getWishList().removePost(post);
            return true;
        } else {
            currentUser.getWishList().addPost(post);
            return false;
        }
    }
}
