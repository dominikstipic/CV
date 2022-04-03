package com.playdate.forms;

import com.playdate.enumerations.PostType;
import lombok.Data;
import org.springframework.web.multipart.MultipartFile;

import javax.validation.constraints.NotEmpty;
import javax.validation.constraints.NotNull;

@Data
public class PostForm {

    @NotEmpty
    private String content;

    private Double latitude;

    private Double longitude;

    @NotNull
    private PostType postType;

    private MultipartFile image;

    private MultipartFile video;


}
