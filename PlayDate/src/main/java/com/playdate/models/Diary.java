package com.playdate.models;

import lombok.Data;
import lombok.EqualsAndHashCode;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import java.util.HashSet;
import java.util.Set;

@Entity
@Table(name = "diaries")
@Data
@EqualsAndHashCode(of = "id")
public class Diary {

    @Id
    private Long id;

    @NotNull
    @OneToOne(fetch = FetchType.LAZY)
    @MapsId
    private User owner;

    @OneToMany(mappedBy = "diary")
    private Set<Post> posts = new HashSet<>();

    public void addPost(Post post) {
        posts.add(post);
        post.setDiary(this);
    }

    public void removePost(Post post) {
        posts.remove(post);
        post.setDiary(null);
    }
}
