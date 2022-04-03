package com.playdate.controllers;

import com.playdate.enumerations.FilterType;
import com.playdate.enumerations.PostType;
import com.playdate.models.User;
import com.playdate.models.dtos.PostForDiary;
import com.playdate.models.dtos.PostForFeed;
import com.playdate.models.dtos.UserProfileInfo;
import com.playdate.services.PostService;
import com.playdate.services.UserService;
import com.playdate.utility.Utility;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.List;
import java.util.Set;

import static com.playdate.utility.Utility.NUMBER_OF_POSTS;

@Controller
public class HomeController {
    private final PostService postService;

    private final UserService userService;

    @Autowired
    public HomeController(PostService postService, UserService userService) {
        this.postService = postService;
        this.userService = userService;
    }

    @GetMapping(value= {"/", "/home", "/index", "/wall", "/feed", "/news"})
    public String home(Model model, @RequestParam(required = false) String filter) {
        User currentUser = userService.getCurrentUser();

        Set<User> users = userService.fetchUserFriends(currentUser.getUsername());
        users.add(currentUser);

        FilterType filterType = FilterType.ALL;

        if (filter != null) {
            filter = filter.trim().toUpperCase();
            if (Utility.validFilterType(filter)) {
                filterType = FilterType.valueOf(filter);
            }
        }

        List<PostForFeed> posts = null;
        if (filterType.equals(FilterType.ALL)) {
            posts = postService.fetchPostsForFeedFromUsers(users, 0, NUMBER_OF_POSTS);
        } else {
            posts = postService.fetchPostsForFeedFromUsersByPostType(users, PostType.valueOf(filterType.toString()), 0, NUMBER_OF_POSTS);
        }

        model.addAttribute("posts", posts);
        model.addAttribute("pageName", Utility.WALL);
        model.addAttribute("filter", filterType.toString());

        return "wall";
    }

    @GetMapping("/activity")
    public String myActivityMap(Model model){
        return "activity_map";
    }

    @GetMapping("/profile")
    public String myProfile(Model model) {
        User currentUser = userService.getCurrentUser();

        List<PostForFeed> posts = postService.fetchPostsForProfileFromUser(currentUser,0, NUMBER_OF_POSTS);

        UserProfileInfo userProfileInfo = userService.getUserProfileInfo(currentUser.getUsername());
        model.addAttribute("userProfileInfo", userProfileInfo);
        model.addAttribute("posts", posts);
        model.addAttribute("pageName", Utility.PROFILE);

        return "wall";
    }

    @GetMapping("/user/{username}")
    public String userProfile(@PathVariable("username") String username, Model model) {
        User user = userService.findByUsername(username);

        List<PostForFeed> posts = postService.fetchPostsForProfileFromUser(user,0, NUMBER_OF_POSTS);

        UserProfileInfo userProfileInfo = userService.getUserProfileInfo(username);

        model.addAttribute("user", user);
        model.addAttribute("userProfileInfo", userProfileInfo);
        model.addAttribute("posts", posts);
        model.addAttribute("pageName", Utility.PROFILE);

        return "wall";
    }

    @GetMapping("/diary")
    public String myDiary(Model model) {
        User currentUser = userService.getCurrentUser();

        List<PostForDiary> posts = postService.fetchPostsForDiaryFromUser(currentUser,0, NUMBER_OF_POSTS);

        model.addAttribute("posts", posts);
        model.addAttribute("pageName", Utility.DIARY);

        return "wall";
    }

    @GetMapping("/user/{username}/diary")
    public String userDiary(@PathVariable("username") String username, Model model) {
        User user = userService.findByUsername(username);

        List<PostForDiary> posts = postService.fetchPostsForDiaryFromUser(user,0, NUMBER_OF_POSTS);

        model.addAttribute("user", user);
        model.addAttribute("posts", posts);
        model.addAttribute("pageName", Utility.DIARY);

        return "wall";
    }

    @GetMapping("/wishlist")
    public String myWishlist(Model model) {
        User currentUser = userService.getCurrentUser();

        List<PostForDiary> posts = postService.fetchPostsForWishlistFromUser(currentUser,0, NUMBER_OF_POSTS);

        model.addAttribute("posts", posts);
        model.addAttribute("pageName", Utility.WISHLIST);

        return "wall";
    }

    @GetMapping("/user/{username}/wishlist")
    public String userWishlist(@PathVariable("username") String username, Model model) {
        User user = userService.findByUsername(username);

        List<PostForDiary> posts = postService.fetchPostsForWishlistFromUser(user,0, NUMBER_OF_POSTS);

        model.addAttribute("user", user);
        model.addAttribute("posts", posts);
        model.addAttribute("pageName", Utility.WISHLIST);

        return "wall";
    }
}
