package com.playdate.controllers;

import com.playdate.enumerations.Gender;
import com.playdate.forms.RegisterForm;
import com.playdate.models.User;
import com.playdate.services.RoleService;
import com.playdate.services.UserService;
import lombok.SneakyThrows;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;

import javax.validation.Valid;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

@Controller
public class RegisterController {

    private final UserService userService;
    private final RoleService roleService;
    private final PasswordEncoder passwordEncoder;

    @Autowired
    public RegisterController(UserService userService, RoleService roleService, PasswordEncoder passwordEncoder) {
        this.userService = userService;
        this.roleService = roleService;
        this.passwordEncoder = passwordEncoder;
    }

    @GetMapping("/register")
    public String index(
            Model model
    ) {
        RegisterForm registerForm = new RegisterForm();
        registerForm.setGender(Gender.FEMALE);

        model.addAttribute("registerForm", registerForm);

        return "register";
    }

    @PostMapping("/register")
    public String register(
            @ModelAttribute @Valid RegisterForm registerForm,
            BindingResult bindingResult) {

        if (bindingResult.hasErrors()) {
            return "register";
        }

        User user = fillUser(registerForm);
        userService.register(user);


        return "redirect:/register?success";
    }

    @SneakyThrows(ParseException.class)
    private User fillUser(RegisterForm registerForm) {
        User user = new User();

        user.setFirstName(registerForm.getFirstName());
        user.setLastName(registerForm.getLastName());

        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("dd/MM/yyyy");
        Date birthday = simpleDateFormat.parse(registerForm.getBirthday());
        user.setBirthday(birthday);
        user.setGender(registerForm.getGender());

        user.setEmail(registerForm.getEmail());
        user.setUsername(registerForm.getUsername());

        user.setPassword(passwordEncoder.encode(registerForm.getPassword()));

        user.addRole(roleService.findRoleByRoleName("USER"));

        return user;
    }

}
