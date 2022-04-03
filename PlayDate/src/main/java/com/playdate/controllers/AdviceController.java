package com.playdate.controllers;

import com.playdate.models.User;
import com.playdate.services.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ModelAttribute;

@ControllerAdvice(assignableTypes = {HomeController.class, CalendarController.class, PostController.class, GroupController.class})
public class AdviceController {

    private final UserService userService;

    @Autowired
    public AdviceController(UserService userService) {
        this.userService = userService;
    }

    @ModelAttribute("currentUser")
    public User getCurrentUser(Model model) {
        return userService.getCurrentUser();
    }
}
