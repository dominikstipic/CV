package com.playdate.validation.validators;

import com.playdate.validation.annotations.ValidDate;
import org.hibernate.validator.internal.constraintvalidators.bv.notempty.NotEmptyValidatorForCharSequence;

import javax.validation.ConstraintValidator;
import javax.validation.ConstraintValidatorContext;
import java.text.ParseException;
import java.text.SimpleDateFormat;

public class ValidDateValidator implements ConstraintValidator<ValidDate, String> {

    @Override
    public boolean isValid(String value, ConstraintValidatorContext context) {
        if (!(new NotEmptyValidatorForCharSequence().isValid(value, context))) {
            return true;
        }

        try {
            new SimpleDateFormat("dd/MM/yyyy").parse(value.toString());

            return true;
        } catch (ParseException ex) {
        }

        return false;
    }
}
