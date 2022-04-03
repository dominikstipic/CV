
package com.playdate.controllers;

import com.playdate.models.dtos.PostForFeed;
import com.playdate.services.PostService;
import com.playdate.services.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Controller
public class PostController {

    private final UserService userService;
    private final PostService postService;

    @Autowired
    public PostController(UserService userService, PostService postService) {
        this.userService = userService;
        this.postService = postService;
    }

    @GetMapping("/post/{postId}")
    public String getSinglePost(Model model, @PathVariable("postId") long postId){
        List<PostForFeed> posts = new ArrayList<>(Arrays.asList(postService.createPostForFeed(postId)));
        model.addAttribute("posts", posts);
        return "wall";
    }
}
