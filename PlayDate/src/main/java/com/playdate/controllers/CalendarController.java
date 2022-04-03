package com.playdate.controllers;

import com.playdate.models.CalendarEntry;
import com.playdate.services.CalendarService;
import com.playdate.services.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.Errors;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.annotation.*;

import java.text.ParseException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Controller
public class CalendarController {
    @Autowired
    private CalendarService calendarService;
    @Autowired
    private UserService userService;
    private String date = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE);



    @GetMapping("/calendar")
    public String getCalendar(Model model){
        Long id = userService.findByUsername(userService.getCurrentUser().getUsername()).getId();
        model.addAttribute("user",userService.getCurrentUser());
        model.addAttribute("entry",new CalendarEntry());
        model.addAttribute("currentPage", date);
        return "calendar";
    }

    @PostMapping("/calendar/{id}")
    public String addCalendarEntry (@PathVariable Long id, @ModelAttribute CalendarEntry entry, Errors errors){
        entry.setId(null);
        System.out.println(entry);
        errorCorrection(entry, errors);
        calendarService.saveEntry(id,entry);
        return "redirect:/calendar";
    }

    @PutMapping("/calendar/{id}")
    public String updateEntryById (@PathVariable Long id, @ModelAttribute CalendarEntry entry,  Errors errors){
        errorCorrection(entry, errors);
        calendarService.updateEntry(id, entry);
        calendarService.getCalendarById(id).getEntries().removeIf(e -> e.getId().equals(id));
        calendarService.getCalendarById(id).addCalendarEntry(entry);
        return "redirect:/calendar";
    }

    @DeleteMapping("/calendar/{id}")
    public String removeEntry(@PathVariable Long id,@ModelAttribute CalendarEntry entry, Errors errors){
        errorCorrection(entry, errors);
        calendarService.deleteEntry(id,entry);
        return "redirect:/calendar";
    }


    @RequestMapping(value = "/calendar/page", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public @ResponseBody void getPageDate(@RequestBody Map<String,String> json) throws ParseException {
        this.date = json.get("date");
    }

    private void errorCorrection(CalendarEntry entry, Errors errors) {
        entry.getStart().setHours(entry.getStart().getHours() - 1);
        entry.getEnd().setHours(entry.getEnd().getHours() - 1);
        if (errors.hasErrors()) {
            List<String> errorList = errors.getFieldErrors().stream().map(FieldError::getField).collect(Collectors.toList());
            if (errorList.contains("pauseFrom") && errorList.contains("pauseTo")) {
                entry.setPauseFrom(null);
                entry.setPauseTo(null);
            }
            if (errorList.contains("repeatInterval")) {
                entry.setRepeatInterval(null);
            }
        }
    }
}

