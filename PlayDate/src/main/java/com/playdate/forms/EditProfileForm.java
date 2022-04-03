package com.playdate.forms;

import lombok.Data;

@Data
public class EditProfileForm {
    private String firstName;
    private String lastName;
    private String phone;
    private String country;
    private String city;
    private String about;
    private String oldEmail;
    private String newEmail;
    private String oldPassword;
    private String newPassword;
    private String confirmPassword;
}
