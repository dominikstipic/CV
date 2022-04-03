package com.playdate.repositories;

import com.playdate.models.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {

    User findByEmailOrUsername(String email, String username);

    User findByEmail(String email);

    Optional<User> findByUsername(String username);
}
