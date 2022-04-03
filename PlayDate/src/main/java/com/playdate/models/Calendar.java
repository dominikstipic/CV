package com.playdate.models;

import lombok.Data;
import lombok.EqualsAndHashCode;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import java.util.HashSet;
import java.util.Set;

@Entity
@Table(name = "calendars")
@Data
@EqualsAndHashCode(of = "id")
public class Calendar {

    @Id
    private Long id;

    @NotNull
    @OneToOne(fetch = FetchType.LAZY)
    @MapsId
    private User owner;

    @OneToMany(mappedBy = "calendar", cascade = CascadeType.ALL, orphanRemoval = true)
    private Set<CalendarEntry> entries = new HashSet<>();

    public void addCalendarEntry(CalendarEntry calendarEntry) {
        entries.add(calendarEntry);
        calendarEntry.setCalendar(this);
    }

    public void removeCalendarEntry(CalendarEntry calendarEntry) {
        entries.remove(calendarEntry);
        calendarEntry.setCalendar(null);
    }
}
