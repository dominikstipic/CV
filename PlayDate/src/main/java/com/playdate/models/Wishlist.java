package com.playdate.models;

import lombok.Data;
import lombok.EqualsAndHashCode;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import java.util.HashSet;
import java.util.Set;

@Entity
@Table(name = "wishlists")
@Data
@EqualsAndHashCode(of = "id")
public class Wishlist {

    @Id
    private Long id;

    @NotNull
    @OneToOne(fetch = FetchType.LAZY)
    @MapsId
    private User owner;

    @ManyToMany(mappedBy = "wishlists", cascade = {CascadeType.PERSIST, CascadeType.MERGE}, fetch = FetchType.LAZY)
    private Set<Post> posts = new HashSet<>();

    public void addPost(Post post) {
        posts.add(post);
        post.getWishlists().add(this);
    }

    public void removePost(Post post) {
        posts.remove(post);
        post.getWishlists().remove(this);
    }
}
