package com.playdate.models.dtos;

import com.playdate.enumerations.PostType;
import lombok.Value;

import java.util.Date;
import java.util.Set;

@Value
public class PostForDiary extends PostForFeed{
    private Boolean isInWishlist;

    public PostForDiary(Long id, UserDto owner, Date datePosted, String content, PostType postType, Set<CommentDto> comments, boolean isLikedByCurrentUser, boolean isResolved, boolean image, boolean video, boolean edited, boolean isInWishlist) {
        super(id, owner, datePosted, content, postType, comments, isLikedByCurrentUser, isResolved, image, video, edited);
        this.isInWishlist = isInWishlist;
    }
}
