package com.playdate.services;

import com.playdate.exception.RequestDeniedException;
import com.playdate.models.Calendar;
import com.playdate.models.CalendarEntry;
import com.playdate.repositories.CalendarRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.util.Assert;

import java.util.Optional;

@Service
public class CalendarService {

    @Autowired
    private CalendarRepository calendarRepository;

    public Calendar getCalendarById(Long id){
        Optional<Calendar> calendar = calendarRepository.findById(id);

        if (!calendar.isPresent()){
            throw new RequestDeniedException("Calendar with given id does not exist");
        }
        return calendar.get();
    }

    public void saveEntry(Long calendarId, CalendarEntry entry) {
        Assert.isNull(entry.getId(), "entry must not have id");
        Calendar c = getCalendarById(calendarId);
        c.addCalendarEntry(entry);
        calendarRepository.save(c);
    }

    public void updateEntry(Long calendarId, CalendarEntry entry){
        Assert.notNull(entry.getId(), "entry id cannot be null");
        if(getCalendarById(calendarId).getEntries().stream().anyMatch(e -> e.getId().equals(calendarId))){
            getCalendarById(calendarId).removeCalendarEntry(entry);
            getCalendarById(calendarId).addCalendarEntry(entry);
            calendarRepository.save(getCalendarById(calendarId));
        }
    }

    public void deleteEntry(Long calendarId, CalendarEntry entry){
        Assert.notNull(entry.getId(), "entry id cannot be null");
        if(getCalendarById(calendarId).getEntries().stream().anyMatch(e -> e.getId().equals(entry.getId()))){
           getCalendarById(calendarId).removeCalendarEntry(entry);
            calendarRepository.save(getCalendarById(calendarId));
        }
    }
}
