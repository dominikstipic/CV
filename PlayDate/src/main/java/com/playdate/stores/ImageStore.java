package com.playdate.stores;

import com.playdate.models.Image;
import org.springframework.content.commons.repository.ContentStore;
import org.springframework.stereotype.Component;

import java.util.UUID;

@Component
public interface ImageStore extends ContentStore<Image, UUID> {
}
