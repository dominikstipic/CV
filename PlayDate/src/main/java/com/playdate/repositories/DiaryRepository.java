package com.playdate.repositories;

import com.playdate.models.Diary;
import com.playdate.models.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface DiaryRepository extends JpaRepository<Diary, Long> {

    Diary findByOwner(User owner);
}
