package com.playdate.models;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.playdate.enumerations.PostType;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;

import javax.persistence.*;
import javax.validation.constraints.NotEmpty;
import javax.validation.constraints.NotNull;
import java.util.Calendar;
import java.util.*;

@Entity
@Table(name = "posts")
@Data
@EqualsAndHashCode(of = "id")
public class Post {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotNull
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "owner_id")
    private User owner;

    @NotNull
    @Enumerated(EnumType.STRING)
    private PostType type;

    @NotNull
    private Boolean closed = false;

    @NotEmpty
    @Lob
    private String content;

    @NotNull
    private Date datePosted = Calendar.getInstance().getTime();

    private Date dateModified;

    @OneToOne(cascade = CascadeType.ALL, orphanRemoval = true)
    @JoinColumn(name = "post_image_id")
    private Image postImage;

    @OneToOne(cascade = CascadeType.ALL, orphanRemoval = true)
    @JoinColumn(name = "post_video_id")
    private Video postVideo;

    @OneToOne(mappedBy = "post", fetch = FetchType.LAZY, cascade = CascadeType.ALL)
    private Location location;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinTable(name = "group_id")
    @ToString.Exclude
    private Group group;

    @ManyToMany(cascade = {CascadeType.PERSIST, CascadeType.MERGE})
    @JoinTable(
            name="post_wishlists",
            joinColumns = @JoinColumn(
                    name="post_id"),
            inverseJoinColumns = @JoinColumn(
                    name="wishlist_id")
    )
    @ToString.Exclude
    private Set<Wishlist> wishlists = new HashSet<>();

    public void addWishlist(Wishlist wishlist) {
        wishlists.add(wishlist);
        wishlist.getPosts().add(this);
    }

    public void removeWishlist(Wishlist wishlist) {
        wishlists.remove(wishlist);
        wishlist.getPosts().remove(this);
    }

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "diary_id")
    @ToString.Exclude
    private Diary diary;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "activity_map_id")
    @ToString.Exclude
    private ActivityMap activityMap;

    @OneToMany(mappedBy = "post", cascade = CascadeType.ALL, orphanRemoval = true)
    private Set<Comment> comments = new LinkedHashSet<>();
    
    public void addComment(Comment comment) {
        comments.add(comment);
        comment.setPost(this);
    }

    public void removeComment(Comment comment) {
        comments.remove(comment);
        comment.setPost(null);
    }

    @ManyToMany(cascade = {CascadeType.PERSIST, CascadeType.MERGE})
    @JoinTable(
            name="post_liked_by",
            joinColumns = @JoinColumn(
                    name="post_id"),
            inverseJoinColumns = @JoinColumn(
                    name="liked_by_id")
    )
    @JsonIgnore
    private Set<User> likedBy = new HashSet<>();

    public void addLikedBy(User user) {
        likedBy.add(user);
        user.getLikedPosts().add(this);
    }

    public void removeLikedBy(User user) {
        likedBy.remove(user);
        user.getLikedPosts().remove(this);
    }
}
