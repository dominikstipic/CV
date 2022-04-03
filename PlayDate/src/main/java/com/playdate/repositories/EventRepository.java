package com.playdate.repositories;

import com.playdate.models.CalendarEntry;
import org.springframework.data.jpa.repository.JpaRepository;

public interface EventRepository extends JpaRepository<CalendarEntry,Long> {

}
