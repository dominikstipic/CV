package com.playdate.models.dtos;

import com.playdate.enumerations.PostType;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.Value;
import lombok.experimental.NonFinal;

import java.util.Date;
import java.util.Set;

@Value
@NonFinal
public class PostForFeed {
    private Long id;
    private UserDto owner;
    private Date datePosted;
    private String content;
    private PostType postType;
    private Set<CommentDto> comments;
    private boolean isLikedByCurrentUser;
    private boolean isResolved;
    private boolean image;
    private boolean video;
    private boolean edited;

    public boolean hasImage() {
        return image;
    }

    public boolean hasVideo() {
        return video;
    }
}
