package com.playdate.controllers;


import com.playdate.models.CalendarEntry;
import com.playdate.services.CalendarService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

import java.util.Collection;

@RestController
public class CalendarRestController {
    @Autowired
    private CalendarService calendarService;

    @GetMapping(value = "/calendar/{id}")
    public Collection<CalendarEntry> getEntries(Model model, @PathVariable Long id) {
        return calendarService.getCalendarById(id).getEntries();
    }



}


