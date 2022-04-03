package com.playdate.services;

import com.playdate.models.Group;
import com.playdate.models.Image;
import com.playdate.models.User;
import com.playdate.repositories.GroupRepository;
import com.playdate.stores.ImageStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

@Service
public class GroupService {
    private final GroupRepository groupRepository;
    private final ImageStore imageStore;

    @Autowired
    public GroupService(GroupRepository groupRepository, ImageStore imageStore){
        this.imageStore = imageStore;
        this.groupRepository = groupRepository;
    }

    public List<Group> findGroupsByMember(User user){
        return this.groupRepository.findAllByMembersContains(user);
    }

    public Group findGroupById(Long id){
        return groupRepository.findByIdIs(id);
    }

    @Transactional
    public Group createGroup(Group group){
        return this.groupRepository.saveAndFlush(group);
    }

    public void addGroupBanner(Group group, MultipartFile image) throws IOException {
        if (image != null) {
            Image img = new Image();
            img.setMimeType(image.getContentType());

            group.setBanner(img);

            imageStore.setContent(img, image.getInputStream());
        }
    }

}
