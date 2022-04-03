package com.playdate.validation.annotations;


import com.playdate.validation.validators.ValidDateValidator;

import javax.validation.Constraint;
import javax.validation.Payload;
import java.lang.annotation.*;

@Documented
@Constraint(validatedBy = ValidDateValidator.class)
@Target({ElementType.FIELD})
@Retention(RetentionPolicy.RUNTIME)
public @interface ValidDate {
    String message() default "{invalid.date.format}";

    Class<?>[] groups() default {};

    Class<? extends Payload>[] payload() default {};
}
