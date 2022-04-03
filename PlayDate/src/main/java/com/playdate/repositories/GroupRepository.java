package com.playdate.repositories;

import com.playdate.models.Group;
import com.playdate.models.Post;
import com.playdate.models.User;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Set;

@Repository
public interface GroupRepository extends JpaRepository<Group, Long> {
    Group findByIdIs(Long id);

    List<Group> findAllByMembersContains(User user);
}
