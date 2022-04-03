package com.playdate.validation.annotations;

import com.playdate.validation.validators.EmailNotTakenValidator;

import javax.validation.Constraint;
import javax.validation.Payload;
import java.lang.annotation.*;

@Documented
@Constraint(validatedBy = EmailNotTakenValidator.class)
@Target({ElementType.FIELD})
@Retention(RetentionPolicy.RUNTIME)
public @interface EmailNotTaken {

    String message() default "{taken.email}";

    Class<?>[] groups() default {};

    Class<? extends Payload>[] payload() default {};
}
