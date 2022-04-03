package com.playdate.controllers;

import com.playdate.models.Group;
import com.playdate.models.User;
import com.playdate.models.dtos.PostForFeed;
import com.playdate.services.GroupService;
import com.playdate.services.PostService;
import com.playdate.services.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static com.playdate.utility.Utility.NUMBER_OF_POSTS;

@Controller
public class GroupController{

    private final UserService userService;
    private final GroupService groupService;
    private final PostService postService;

    @Autowired
    public GroupController(GroupService groupService, UserService userService, PostService postService) {
        this.userService = userService;
        this.groupService = groupService;
        this.postService = postService;
    }

    @ModelAttribute("groupForm")
    public Group getGroupForm(){
        return new Group();
    }

    @GetMapping("/groups")
    public String getUserGroups(Model model){
        User currentUser = this.userService.getCurrentUser();

        List<Group> groups = this.groupService.findGroupsByMember(currentUser);
        model.addAttribute("groups", groups);

        return "groups";
    }

    @PostMapping("/group/create")
    public String createGroup(@Valid @ModelAttribute("groupForm") Group group, BindingResult result, Model model){
        User currentUser = this.userService.getCurrentUser();

        if(result.hasErrors()){

            List<Group> groups = this.groupService.findGroupsByMember(currentUser);
            model.addAttribute("groups", groups);
            model.addAttribute("groupForm", group);

            return "groups";
        }
        else {

            group.setAdmin(currentUser);
            if(group.getImage().getSize() != 0){
                try {
                    groupService.addGroupBanner(group, group.getImage());
                }
                catch(IOException e){
                }
            }
            currentUser.addMemberGroup(group);

            groupService.createGroup(group);
        }
        return "redirect:/group/" + group.getId();
        }

    @GetMapping("/group/{groupId}")
    public String getSpecificGroup(@PathVariable(value="groupId") Long id, Model model){
        User currentUser = this.userService.getCurrentUser();
        Group group = groupService.findGroupById(id);

        List<PostForFeed> posts = new ArrayList<>();
        Set<User> friends = currentUser.getFriends();
        Set<User> suggestedMembers = new HashSet<>();

        for(User friend : friends){
            if(!friend.getGroupsMember().contains(group)){
                suggestedMembers.add(friend);
                if(suggestedMembers.size() > 3){
                    break;
                }
            }
        }

        if(group.getMembers().contains(currentUser)){
            posts = postService.fetchPostsForFeedFromGroup(group, 0, NUMBER_OF_POSTS);
        }

        model.addAttribute("posts", posts);
        model.addAttribute("groupId", id);
        model.addAttribute("group", group);
        model.addAttribute("numOfFollowers", group.getMembers().size());
        model.addAttribute("suggestedMembers", suggestedMembers);

        return "group";
    }

    @PostMapping("/group/{groupId}/add/{username}")
    public String addUserToGroup(@PathVariable("username") String username, @PathVariable("groupId") Long id){
        User user = this.userService.findByUsername(username);
        Group group = this.groupService.findGroupById(id);

        this.userService.joinGroup(group, user);
        return "redirect:/group/" + id;
    }
}
