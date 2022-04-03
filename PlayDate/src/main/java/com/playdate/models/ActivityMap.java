package com.playdate.models;

import lombok.Data;
import lombok.EqualsAndHashCode;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import java.util.HashSet;
import java.util.Set;

@Entity
@Table(name = "activity_maps")
@Data
@EqualsAndHashCode(of = "id")
public class ActivityMap {
    @Id
    private Long id;

    @NotNull
    @OneToOne(fetch = FetchType.LAZY)
    @MapsId
    private User owner;

    @OneToMany(mappedBy = "activityMap")
    private Set<Post> posts = new HashSet<>();

    public void addPost(Post post) {
        if (post.getLocation() == null) {
            throw new IllegalArgumentException("PostDto without location cannot be put onto the activity map!");
        }
        posts.add(post);
        post.setActivityMap(this);
    }

    public void removePost(Post post) {
        posts.remove(post);
        post.setActivityMap(null);
    }
}
