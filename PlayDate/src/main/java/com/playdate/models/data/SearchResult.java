package com.playdate.models.data;

import com.playdate.models.dtos.GroupDto;
import com.playdate.models.dtos.UserDto;
import lombok.Value;

import java.util.List;

@Value
public class SearchResult {
    private List<UserDto> users;
    private List<GroupDto> groups;
}
