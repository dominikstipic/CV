package com.playdate.controllers;

import com.playdate.models.Group;
import com.playdate.models.User;
import com.playdate.models.data.SearchResult;
import com.playdate.models.dtos.GroupDto;
import com.playdate.models.dtos.UserDto;
import com.playdate.services.GroupService;
import com.playdate.services.SearchService;
import com.playdate.services.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
public class SearchRestController {

    private final SearchService searchService;
    private final GroupService groupService;
    private final UserService userService;

    @Autowired
    public SearchRestController(SearchService searchService, GroupService groupService, UserService userService) {
        this.searchService = searchService;
        this.groupService = groupService;
        this.userService = userService;
    }

    @GetMapping("/search")
    public SearchResult search(@RequestParam("query") String query) {
        List<UserDto> users = searchService.searchUsers(query);
        List<GroupDto> groups = searchService.searchGroups(query);

        return new SearchResult(users, groups);
    }

    @GetMapping("/search/group/{groupId}")
    public List<UserDto> searchFriendsNotInGroup(@PathVariable("groupId") Long id, @RequestParam("query") String query){
        Group group = this.groupService.findGroupById(id);
        User currentUser = this.userService.getCurrentUser();

        List<UserDto> users = searchService.searchFriendsNotInGroup(query, group, currentUser);
        return users;
    }
}
