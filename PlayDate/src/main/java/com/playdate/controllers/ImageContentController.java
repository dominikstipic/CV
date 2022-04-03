package com.playdate.controllers;

import com.playdate.models.Group;
import com.playdate.models.Post;
import com.playdate.models.User;
import com.playdate.repositories.GroupRepository;
import com.playdate.repositories.PostRepository;
import com.playdate.repositories.UserRepository;
import com.playdate.stores.ImageStore;
import lombok.SneakyThrows;
import org.apache.commons.io.IOUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.io.InputStream;
import java.util.Optional;

@RestController
public class ImageContentController {

    @Autowired
    private ImageStore imageStore;

    @Autowired
    private PostRepository postRepository;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private GroupRepository groupRepository;

    @GetMapping("/post/{postId}/image")
    public ResponseEntity<?> getPostImage(@PathVariable("postId") Long id) {
        Optional<Post> post = postRepository.findById(id);

        if (post.isPresent()) {
            Post post1 = post.get();
            InputStreamResource inputStreamResource = new InputStreamResource(imageStore.getContent(post1.getPostImage()));
            HttpHeaders headers = new HttpHeaders();
            headers.setContentLength(post1.getPostImage().getContentLength());
            headers.set("Content-Type", post1.getPostImage().getMimeType());

            return new ResponseEntity<Object>(inputStreamResource, headers, HttpStatus.OK);
        }

        return null;
    }

    @GetMapping("/users/{username}/image")
    @SneakyThrows(IOException.class)
    public ResponseEntity<?> getUserImage(@PathVariable("username") String username) {
        Optional<User> userOptional = userRepository.findByUsername(username);

        HttpHeaders headers = new HttpHeaders();
        if (userOptional.isPresent() && userOptional.get().getProfilePicture() != null) {
            User user = userOptional.get();
            InputStreamResource inputStreamResource = new InputStreamResource(imageStore.getContent(user.getProfilePicture()));
            headers.setContentLength(user.getProfilePicture().getContentLength());
            headers.set("Content-Type", user.getProfilePicture().getMimeType());

            return new ResponseEntity<Object>(inputStreamResource, headers, HttpStatus.OK);
        }
        InputStream in = getClass().getResourceAsStream("/static/img/default_profile_picture.jpg");
        headers.set("Content-Type", "image/jpeg");

        return new ResponseEntity<Object>(IOUtils.toByteArray(in), HttpStatus.OK);
    }

    @GetMapping("/groups/{id}/banner")
    @SneakyThrows(IOException.class)
    public ResponseEntity<?> getGroupBanner(@PathVariable("id") Long id) {
        Optional<Group> group = groupRepository.findById(id);

        HttpHeaders headers = new HttpHeaders();
        if (group.isPresent() && group.get().getBanner() != null) {
            Group g = group.get();
            InputStreamResource inputStreamResource = new InputStreamResource(imageStore.getContent(g.getBanner()));
            headers.setContentLength(g.getBanner().getContentLength());
            headers.set("Content-Type", g.getBanner().getMimeType());

            return new ResponseEntity<Object>(inputStreamResource, headers, HttpStatus.OK);
        }
        InputStream in = getClass().getResourceAsStream("/static/img/default_group_cover_photo.jpg");
        headers.set("Content-Type", "image/jpeg");

        return new ResponseEntity<Object>(IOUtils.toByteArray(in), HttpStatus.OK);
    }
}
