package com.playdate.validation.validators;

import com.playdate.services.UserService;
import com.playdate.validation.annotations.EmailNotTaken;
import org.hibernate.validator.internal.constraintvalidators.bv.EmailValidator;
import org.hibernate.validator.internal.constraintvalidators.bv.notempty.NotEmptyValidatorForCharSequence;
import org.springframework.beans.factory.annotation.Autowired;

import javax.validation.ConstraintValidator;
import javax.validation.ConstraintValidatorContext;

public class EmailNotTakenValidator implements ConstraintValidator<EmailNotTaken, String> {

    @Autowired
    private UserService userService;

    @Override
    public boolean isValid(String email, ConstraintValidatorContext context) {
        if (!(new NotEmptyValidatorForCharSequence().isValid(email, context)) || !(new EmailValidator()).isValid(email, context)) {
            return true;
        }

        return userService.findByEmail(email) == null;
    }
}
