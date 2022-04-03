package com.playdate.repositories;

import com.playdate.enumerations.PostType;
import com.playdate.models.Diary;
import com.playdate.models.Group;
import com.playdate.models.Post;
import com.playdate.models.User;
import com.playdate.models.Wishlist;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Set;

@Repository
public interface PostRepository extends JpaRepository<Post, Long> {

    List<Post> findAllByOrderByDatePostedDesc();

    Page<Post> findByOwnerIsIn(Set<User> users, Pageable pageable);

    Page<Post> findByGroupEquals(Group group, Pageable pageable);

    Page<Post> findByOwnerIsInAndType(Set<User> users, PostType type, Pageable pageable);

    Page<Post> findByDiaryIsIn(Diary diary, Pageable pageable);

    Page<Post> findByWishlistsIsIn(Wishlist wishlist, Pageable pageable);
}

