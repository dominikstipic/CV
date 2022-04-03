package com.playdate.models.dtos;

import lombok.Value;
import lombok.experimental.NonFinal;

@Value
@NonFinal
public class UserDto {
    private String firstName;
    private String lastName;
    private String username;
    private String email;
    private Long id;
}
