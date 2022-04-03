package com.playdate.stores;

import com.playdate.models.Video;
import org.springframework.content.commons.repository.ContentStore;
import org.springframework.stereotype.Component;

import java.util.UUID;

@Component
public interface VideoStore extends ContentStore<Video, UUID> {
}
