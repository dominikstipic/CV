package com.playdate.models.dtos;

import lombok.Value;

@Value
public class EditProfileInfo {
    private String firstName;
    private String lastName;
    private String email;
    private String phone;
    private String country;
    private String city;
    private String about;
    private boolean isEmailUsed;
    private boolean isPasswordValid;
}
