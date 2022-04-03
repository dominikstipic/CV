package com.playdate.models.dtos;

import com.playdate.enumerations.PostType;
import lombok.Value;

@Value
public class EditPostInfo {
    private Long id;
    private String content;
    private Double longitude;
    private Double latitude;
    private PostType postType;
    private boolean image;
    private boolean video;
}
