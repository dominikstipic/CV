package com.playdate.models;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.springframework.content.commons.annotations.ContentId;
import org.springframework.content.commons.annotations.ContentLength;
import org.springframework.content.commons.annotations.MimeType;

import javax.persistence.*;
import java.util.UUID;

@Entity
@Table(name = "images")
@Data
@EqualsAndHashCode(of = "id")
public class Image {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @OneToOne(mappedBy = "profilePicture")
    private User user;

    @OneToOne(mappedBy = "postImage")
    private Post post;

    @OneToOne(mappedBy = "banner")
    private Group group;

    @ContentId
    private UUID contentId;

    @ContentLength
    private Long contentLength;

    @MimeType
    private String mimeType;
}
