package com.playdate.models;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.hibernate.search.annotations.Field;
import org.hibernate.search.annotations.Indexed;
import org.springframework.web.multipart.MultipartFile;

import javax.persistence.*;
import javax.validation.constraints.NotEmpty;
import javax.validation.constraints.NotNull;
import java.util.HashSet;
import java.util.Set;

@Entity
@Table(name = "groups")
@Data
@EqualsAndHashCode(of = "id")
@Indexed
public class Group {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotEmpty(message = "Group name cannot be empty.")
    @Field
    private String groupName;

    @NotEmpty(message = "Description cannot be empty.")
    private String description;

    @NotNull
    private Boolean isPrivate = false;

    @Transient
    MultipartFile image;

    @OneToOne(cascade = CascadeType.ALL, orphanRemoval = true)
    @JoinColumn(name = "banner_id")
    private Image banner;

    @ManyToOne
    @JoinColumn(name = "admin_id")
    private User admin;

    @ManyToMany(mappedBy = "groupsMember", cascade = {CascadeType.PERSIST, CascadeType.MERGE})
    private Set<User> members = new HashSet<>();

    @OneToMany(mappedBy = "group", fetch = FetchType.LAZY)
    private Set<Post> posts = new HashSet<>();

    public void addPost(Post post) {
        posts.add(post);
        post.setGroup(this);
    }

    public void removePost(Post post) {
        posts.remove(post);
        post.setGroup(null);
    }
}
