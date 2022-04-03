package com.playdate.models;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.playdate.enumerations.Gender;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import org.hibernate.search.annotations.Field;
import org.hibernate.search.annotations.Indexed;

import javax.persistence.*;
import javax.validation.constraints.Email;
import javax.validation.constraints.NotEmpty;
import javax.validation.constraints.NotNull;
import java.util.Date;
import java.util.HashSet;
import java.util.Set;

@Entity
@Table(name = "users")
@Data
@EqualsAndHashCode(of = "username")
@Indexed
public class User {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotEmpty
    @Field
    private String firstName;

    @NotEmpty
    @Field
    private String lastName;

    @NotEmpty
    @Column(unique = true)
    private String username;

    @NotNull
    @Enumerated(EnumType.STRING)
    private Gender gender;

    @NotEmpty
    private String password;

    @NotEmpty
    @Email
    @Column(unique = true)
    private String email;

    @NotNull
    private Date birthday;

    @NotNull
    private Boolean enabled = true;

    @NotNull
    private Boolean credentialsExpired = false;

    @NotNull
    private Boolean expired = false;

    @NotNull
    private Boolean locked = false;

    private String phone;

    private String country;

    private String city;

    @Column(length = 65535, columnDefinition = "text")
    private String about;

    @Transient
    @Field
    public String getFullName() {
        return firstName + " " + lastName;
    }

    @OneToOne(cascade = CascadeType.ALL, orphanRemoval = true)
    @JoinColumn(name = "profile_picture_id")
    private Image profilePicture;

    @ManyToMany(
            fetch = FetchType.LAZY,
            cascade = {CascadeType.PERSIST, CascadeType.MERGE}
    )
    @JoinTable(
            name = "user_roles",
            joinColumns = @JoinColumn(
                    name = "user_id"),
            inverseJoinColumns = @JoinColumn(
                    name = "role_id")
    )
    private Set<Role> roles = new HashSet<>();

    public void addRole(Role role) {
        roles.add(role);
        role.getUsers().add(this);
    }

    public void removeRole(Role role) {
        roles.remove(role);
        role.getUsers().remove(this);
    }

    @ManyToMany(fetch = FetchType.LAZY, cascade = {CascadeType.PERSIST, CascadeType.MERGE})
    @JoinTable(
            name="user_friends",
            joinColumns = @JoinColumn(
                    name="user_id"),
            inverseJoinColumns = @JoinColumn(
                    name="friend_id")
    )
    @ToString.Exclude
    private Set<User> friends = new HashSet<>();

    /**
     * This user is admin in those groups
     */
    @OneToMany(mappedBy = "admin", fetch = FetchType.LAZY, cascade = CascadeType.ALL, orphanRemoval = true)
    @ToString.Exclude
    private Set<Group> groupsAdmin = new HashSet<>();

    public void addAdminGroup(Group group) {
        groupsAdmin.add(group);
        group.setAdmin(this);
    }
    public void removeAdminGroup(Group group) {
        groupsAdmin.remove(group);
        group.setAdmin(null);
    }

    /**
     * This user is a member of those groups
     */
    @ManyToMany(fetch = FetchType.LAZY, cascade = {CascadeType.PERSIST, CascadeType.MERGE})
    @JoinTable(
            name="user_groups",
            joinColumns = @JoinColumn(
                    name="user_id"),
            inverseJoinColumns = @JoinColumn(
                    name="group_id")
    )
    @ToString.Exclude
    private Set<Group> groupsMember = new HashSet<>();

    public void addMemberGroup(Group group) {
        groupsMember.add(group);
        group.getMembers().add(this);
    }

    public void removeMemberGroup(Group group) {
        groupsMember.remove(group);
        group.getMembers().remove(this);
    }

    @OneToMany(mappedBy = "owner", cascade = CascadeType.ALL, orphanRemoval = true)
    @ToString.Exclude
    private Set<Post> posts = new HashSet<>();

    public void addPost(Post post) {
        posts.add(post);
        post.setOwner(this);
    }

    public void removePost(Post post) {
        posts.remove(post);
        post.setOwner(null);
    }

    @ManyToMany(mappedBy = "likedBy", cascade = {CascadeType.PERSIST, CascadeType.MERGE})
    @JsonIgnore
    private Set<Post> likedPosts = new HashSet<>();

    @OneToOne(mappedBy = "owner", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.LAZY)
    @ToString.Exclude
    private Wishlist wishList;

    @OneToOne(mappedBy = "owner", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.LAZY)
    @ToString.Exclude
    private ActivityMap activityMap;

    @OneToOne(mappedBy = "owner", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.LAZY)
    @ToString.Exclude
    private Diary diary;

    @OneToOne(mappedBy = "owner", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.LAZY)
    @ToString.Exclude
    private Calendar calendar;
}
