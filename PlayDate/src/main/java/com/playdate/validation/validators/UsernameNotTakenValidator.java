package com.playdate.validation.validators;

import com.playdate.services.UserService;
import com.playdate.validation.annotations.UsernameNotTaken;
import org.hibernate.validator.internal.constraintvalidators.bv.notempty.NotEmptyValidatorForCharSequence;
import org.springframework.beans.factory.annotation.Autowired;

import javax.validation.ConstraintValidator;
import javax.validation.ConstraintValidatorContext;

public class UsernameNotTakenValidator implements ConstraintValidator<UsernameNotTaken, String> {

    @Autowired
    private UserService userService;

    @Override
    public boolean isValid(String username, ConstraintValidatorContext context) {
        if (!(new NotEmptyValidatorForCharSequence().isValid(username, context))) {
            return true;
        }

        return userService.findByUsername(username) == null;
    }
}
