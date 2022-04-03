package com.playdate.utility;

import com.playdate.enumerations.FilterType;
import com.playdate.enumerations.PostType;
import com.playdate.models.dtos.PostForFeed;
import org.joda.time.Days;
import org.joda.time.LocalDate;
import org.joda.time.Years;
import org.ocpsoft.prettytime.PrettyTime;

import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Predicate;

import static com.playdate.enumerations.FilterType.*;

public class Utility {
    public static final int NUMBER_OF_POSTS = 4;
    public static final int MAX_POST_AGE = 3;   //IN DAYS

    private static final PrettyTime PRETTY_TIME = new PrettyTime();

    public static String formatPrettyDate(Date date) {
        return PRETTY_TIME.format(date);
    }

    public static final String WALL = "WALL";
    public static final String PROFILE = "PROFILE";
    public static final String WISHLIST = "WISHLIST";
    public static final String DIARY = "DIARY";

    public static final Map<FilterType, Predicate<PostForFeed>> postForFeedPredicateMap;

    static {
        postForFeedPredicateMap = new HashMap<>();
        postForFeedPredicateMap.put(ALL, p -> true);
        postForFeedPredicateMap.put(TEXT, (p) -> p.getPostType().equals(PostType.TEXT));
        postForFeedPredicateMap.put(DEMAND, (p) -> p.getPostType().equals(PostType.DEMAND));
        postForFeedPredicateMap.put(OFFER, (p) -> p.getPostType().equals(PostType.OFFER));
    }

    public static Predicate<PostForFeed> getPredicateForString(String filter) {
        if(filter == null) return postForFeedPredicateMap.get(ALL);

        filter = filter.trim().toUpperCase();

        if(!validFilterType(filter)) return postForFeedPredicateMap.get(ALL);

        return postForFeedPredicateMap.get(FilterType.valueOf(filter));
    }

    public static boolean validPhoneNumber(String phoneNumber) {

        return phoneNumber.matches("^[+]?[(]?[0-9]{3}[)]?[-\\s.]?[0-9]{3}[-\\s.]?[0-9]{4,6}$");
    }

    public static Integer calculateAge(Date birthday) {
        return Years.yearsBetween(new LocalDate(birthday), new LocalDate()).getYears();
    }

    public static Integer calcualteDays(Date date) {
        return Days.daysBetween(new LocalDate(date), new LocalDate()).getDays();
    }

    public static boolean validFilterType(String filter) {
        for (FilterType filterType : values()) {
            if (filterType.toString().equals(filter)) {
                return true;
            }
        }
        return false;
    }

    public static FilterType getFilterTypeForString(String filter) {
        if(filter == null) return ALL;

        filter = filter.trim().toUpperCase();

        if(!validFilterType(filter)) return ALL;

        return FilterType.valueOf(filter);
    }
}
