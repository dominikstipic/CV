package com.playdate.models.dtos;

import lombok.Value;

import java.util.Date;

@Value
public class PostForActivityMap {
    private Long postId;
    private UserDto owner;
    private String content;
    private Double latitude;
    private Double longitude;
    private Date datePosted;
}
