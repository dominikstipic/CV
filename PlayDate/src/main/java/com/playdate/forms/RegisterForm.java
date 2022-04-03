package com.playdate.forms;

import com.playdate.enumerations.Gender;
import com.playdate.validation.annotations.*;
import lombok.Data;
import lombok.EqualsAndHashCode;

import javax.validation.constraints.Email;
import javax.validation.constraints.NotEmpty;
import javax.validation.constraints.NotNull;

@PasswordsMatch(first = "password", second = "confirmPassword", message = "The password fields must match")
@Data
@EqualsAndHashCode(of = "username")
public class RegisterForm {

    @NotEmpty(message = "First name is required")
    private String firstName;

    @NotEmpty(message = "Last name is required")
    private String lastName;

    @NotEmpty(message = "Username is required")
    @UsernameNotTaken
    private String username;

    @NotNull
    private Gender gender;

    @NotEmpty(message = "Password is required")
    private String password;

    private String confirmPassword;

    @NotEmpty(message = "Email is required")
    @Email(message = "Invalid email address")
    @EmailNotTaken
    private String email;

    @NotEmpty(message = "Birthday is required")
    @ValidDate
    @LegalAge
    private String birthday;
}
