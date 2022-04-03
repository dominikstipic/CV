package com.playdate.validation.validators;

import com.playdate.validation.annotations.LegalAge;
import lombok.SneakyThrows;
import org.hibernate.validator.internal.constraintvalidators.bv.notempty.NotEmptyValidatorForCharSequence;

import javax.validation.ConstraintValidator;
import javax.validation.ConstraintValidatorContext;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.LocalDate;
import java.time.Period;
import java.time.ZoneId;

public class LegalAgeValidator implements ConstraintValidator<LegalAge, String> {

    @SneakyThrows(ParseException.class)
    @Override
    public boolean isValid(String value, ConstraintValidatorContext context) {
        if (!(new NotEmptyValidatorForCharSequence().isValid(value, context))) {
            return true;
        }
        if (!(new ValidDateValidator().isValid(value, context))) {
            return true;
        }

        return Period.between(new SimpleDateFormat("dd/MM/yyyy").parse(value).toInstant().atZone(ZoneId.systemDefault()).toLocalDate(), LocalDate.now()).getYears() >= 18;
    }
}
