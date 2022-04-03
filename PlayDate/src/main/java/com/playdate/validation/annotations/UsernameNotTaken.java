package com.playdate.validation.annotations;

import com.playdate.validation.validators.UsernameNotTakenValidator;

import javax.validation.Constraint;
import javax.validation.Payload;
import java.lang.annotation.*;

@Documented
@Constraint(validatedBy = UsernameNotTakenValidator.class)
@Target({ElementType.FIELD})
@Retention(RetentionPolicy.RUNTIME)
public @interface UsernameNotTaken {
    String message() default "{taken.username}";

    Class<?>[] groups() default {};

    Class<? extends Payload>[] payload() default {};

}
