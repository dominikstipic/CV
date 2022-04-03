package com.playdate.models.dtos;

import com.playdate.enumerations.Gender;
import lombok.Value;

@Value
public class UserProfileInfo extends UserDto {
    private String phone;
    private Integer age;
    private Gender gender;
    private String country;
    private String city;
    private String about;
    private Boolean userFollows;
    private Boolean banned;

    public UserProfileInfo(String firstName, String lastName, String username, String email,
                           String phone, Integer age, Gender gender, String country, String city, String about, Boolean userFollows, Boolean banned) {
        super(firstName, lastName, username, email,null);
        this.phone = phone;
        this.age = age;
        this.gender = gender;
        this.country = country;
        this.city = city;
        this.about = about;
        this.userFollows = userFollows;
        this.banned = banned;
    }

}
