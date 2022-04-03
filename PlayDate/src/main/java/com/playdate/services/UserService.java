package com.playdate.services;

import com.playdate.forms.EditProfileForm;
import com.playdate.models.*;
import com.playdate.models.dtos.EditProfileInfo;
import com.playdate.models.dtos.UserProfileInfo;
import com.playdate.repositories.UserRepository;
import com.playdate.stores.ImageStore;
import com.playdate.utility.Utility;
import lombok.SneakyThrows;
import org.apache.commons.validator.EmailValidator;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Optional;
import java.util.Set;


@Service
@Transactional
public class UserService {

    private final UserRepository userRepository;

    private final PasswordEncoder passwordEncoder;

    private final ImageStore imageStore;

    @Autowired
    public UserService(UserRepository userRepository, PasswordEncoder passwordEncoder, ImageStore imageStore) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
        this.imageStore = imageStore;
    }

    public User findByEmailOrUsername(String email, String username) {
        return userRepository.findByEmailOrUsername(email, username);
    }

    public User findByEmail(String email) {
        return userRepository.findByEmail(email);
    }

    public User findByUsername(String username) {
        Optional<User> userOptional = userRepository.findByUsername(username);
        return userOptional.isPresent() ? userOptional.get() : null;
    }

    public User register(User user) {

        user = userRepository.save(user);

        ActivityMap activityMap = new ActivityMap();
        activityMap.setOwner(user);
        Calendar calendar = new Calendar();
        calendar.setOwner(user);
        Diary diary = new Diary();
        diary.setOwner(user);
        Wishlist wishList = new Wishlist();
        wishList.setOwner(user);

        user.setActivityMap(activityMap);
        user.setCalendar(calendar);
        user.setDiary(diary);
        user.setWishList(wishList);

        userRepository.flush();

        return  user;
    }

    public User getCurrentUser() {
        Optional<User> userOptional = userRepository.findByUsername(((org.springframework.security.core.userdetails.User) SecurityContextHolder.getContext().getAuthentication().getPrincipal()).getUsername());
        return userOptional.isPresent() ? userOptional.get() : null;
    }

    @Transactional
    public Set<User> fetchUserFriends(String username) {
        return findByUsername(username).getFriends();
    }

    @Transactional
    public UserProfileInfo getUserProfileInfo(String username) {
        User user = findByUsername(username);

        return user != null ? new UserProfileInfo(
                user.getFirstName(),
                user.getLastName(),
                user.getUsername(),
                user.getEmail(),
               user.getPhone(),
                Utility.calculateAge(user.getBirthday()),
                user.getGender(), user.getCountry(),
                user.getCity(), user.getAbout(),
                getCurrentUser().getFriends().contains(user),
                !user.getEnabled()) : null;
    }

    @Transactional
    public EditProfileInfo getUserEditProfileInfo(EditProfileForm editProfileForm) {
        User currentUser = getCurrentUser();

        return new EditProfileInfo(
                currentUser.getFirstName(),
                currentUser.getLastName(),
                currentUser.getEmail(),
                currentUser.getPhone(),
                currentUser.getCountry(),
                currentUser.getCity(),
                currentUser.getAbout(),
                userRepository.findByEmail(editProfileForm.getNewEmail()) != null,
                passwordEncoder.matches(editProfileForm.getOldPassword(), currentUser.getPassword()));
    }

    @Transactional
    public void editUser(EditProfileForm editProfileForm) {
        User u = getCurrentUser();
        String firstName = editProfileForm.getFirstName();
        if (firstName != null && firstName.length() > 2 && firstName.length() < 30) {
            u.setFirstName(firstName);
        }
        String lastName = editProfileForm.getLastName();
        if (lastName != null && lastName.length() > 2 && lastName.length() < 30) {
            u.setLastName(lastName);
        }

        String phone = editProfileForm.getPhone();
        if (phone != null) {
            if (phone.isEmpty()) {
                u.setPhone(null);
            } else if (Utility.validPhoneNumber(phone)) {
                u.setPhone(phone);
            }
        }

        String country = editProfileForm.getCountry();
        if (country != null) {
            if (country.isEmpty()) {
                u.setCountry(null);
            } else {
                u.setCountry(country);
            }
        }

        String city = editProfileForm.getCity();
        if (city != null) {
            if (city.isEmpty()) {
                u.setCity(null);
            } else {
                u.setCity(city);
            }
        }

        String about = editProfileForm.getAbout();
        if (about != null) {
            if (about.isEmpty()) {
                u.setAbout(null);
            } else {
                u.setAbout(about);
            }
        }

        String oldEmail = editProfileForm.getOldEmail();
        String newEmail = editProfileForm.getNewEmail();
        if (oldEmail != null && newEmail != null && oldEmail.equals(u.getEmail()) && EmailValidator.getInstance().isValid(newEmail) && findByEmail(newEmail) == null) {
            u.setEmail(editProfileForm.getNewEmail());
        }

        String oldPassword = editProfileForm.getOldPassword();
        String newPassword = editProfileForm.getNewPassword();
        String confirmPassword = editProfileForm.getConfirmPassword();

        if (oldPassword != null && passwordEncoder.matches(oldPassword, u.getPassword()) && newPassword != null && newPassword.equals(confirmPassword)) {
            u.setPassword(passwordEncoder.encode(newPassword));
        }

        userRepository.flush();
    }

    @Transactional
    @SneakyThrows(IOException.class)
    public void updateProfilePicture(MultipartFile image) {
        User currentUser = getCurrentUser();

        if (image != null) {
            Image img = new Image();
            img.setMimeType(image.getContentType());

            img.setUser(currentUser);
            currentUser.setProfilePicture(img);

            imageStore.setContent(img, image.getInputStream());
        }
    }

    @Transactional
    public void followUser(Long userId) {
        Optional<User> u = userRepository.findById(userId);

        u.ifPresent(usr -> {
            User currentUser = getCurrentUser();
            currentUser.getFriends().add(usr);
            userRepository.save(currentUser);
        });
    }

    @Transactional
    public void unfollowUser(Long userId) {
        Optional<User> u = userRepository.findById(userId);

        u.ifPresent(usr -> {
            User currentUser = getCurrentUser();
            currentUser.getFriends().remove(usr);
            userRepository.save(currentUser);
        });
    }

    @Transactional
    public void banUser(Long userId) {
        Optional<User> u = userRepository.findById(userId);

        u.ifPresent(usr -> {
            usr.setEnabled(false);
            userRepository.save(usr);
        });
    }

    @Transactional
    public void unbanUser(Long userId) {
        Optional<User> u = userRepository.findById(userId);

        u.ifPresent(usr -> {
            usr.setEnabled(true);
            userRepository.save(usr);
        });
    }

    @Transactional
    public void joinGroup(Group group, User user) {
        user.getGroupsMember().add(group);
        this.userRepository.save(user);
    }
}
