package com.playdate.controllers;

import com.playdate.models.Post;
import com.playdate.models.User;
import com.playdate.models.Video;
import com.playdate.repositories.PostRepository;
import com.playdate.repositories.UserRepository;
import com.playdate.stores.ImageStore;
import com.playdate.stores.VideoStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

import java.util.Optional;

@RestController
public class VideoContentController {

    @Autowired
    private VideoStore videoStore;

    @Autowired
    private PostRepository postRepository;


    @GetMapping("/post/{postId}/video")
    public ResponseEntity<?> getPostImage(@PathVariable("postId") Long id) {
        Optional<Post> post = postRepository.findById(id);

        if (post.isPresent()) {
            Post post1 = post.get();
            Video postVideo = post1.getPostVideo();

            InputStreamResource inputStreamResource = new InputStreamResource(videoStore.getContent(postVideo));

            HttpHeaders headers = new HttpHeaders();
            headers.setContentLength(postVideo.getContentLength());
            headers.set("Content-Type", postVideo.getMimeType());

            return new ResponseEntity<Object>(inputStreamResource, headers, HttpStatus.OK);
        }

        return null;
    }
}
