package com.playdate.models.dtos;

import lombok.Value;

@Value
public class CommentDto {
    private UserDto owner;
    private String content;
}
