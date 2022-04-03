package com.playdate.controllers;

import com.playdate.forms.EditProfileForm;
import com.playdate.models.dtos.EditProfileInfo;
import com.playdate.services.UserService;
import lombok.SneakyThrows;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.annotation.Secured;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

@RestController
public class UserRestController {

    private final UserService userService;

    private final ImageContentController imageContentController;

    @Autowired
    public UserRestController(UserService userService, ImageContentController imageContentController) {
        this.userService = userService;
        this.imageContentController = imageContentController;
    }

    @PostMapping("/profile/edit/information")
    public EditProfileInfo myEditProfileInformation(
            @RequestBody EditProfileForm editProfileForm
    ) {
        return userService.getUserEditProfileInfo(editProfileForm);
    }

    @PostMapping("/profile/edit")
    public void editProfile(
            @RequestBody EditProfileForm editProfileForm
    ) {
        userService.editUser(editProfileForm);
    }

    @PostMapping("/profile/edit/avatar")
    @SneakyThrows(IOException.class)
    public void editProfilePicture(
            HttpServletResponse response,
            @RequestParam(value = "image") MultipartFile image
    ) {
        userService.updateProfilePicture(image);

        response.sendRedirect("/profile");
    }

    @PostMapping("/user/{userId}/follow")
    public void followUser(@PathVariable("userId") Long userId) {
        userService.followUser(userId);
    }

    @PostMapping("/user/{userId}/unfollow")
    public void unfollowUser(@PathVariable("userId") Long userId) {
        userService.unfollowUser(userId);
    }

    @PostMapping("/user/{userId}/ban")
    @Secured("ROLE_ADMIN")
    public void banUser(@PathVariable("userId") Long userId) {
        userService.banUser(userId);
    }

    @PostMapping("/user/{userId}/unban")
    @Secured("ROLE_ADMIN")
    public void unbanUser(@PathVariable("userId") Long userId) {
        userService.unbanUser(userId);
    }
}
