package com.playdate.validation.annotations;

import com.playdate.validation.validators.LegalAgeValidator;

import javax.validation.Constraint;
import javax.validation.Payload;
import java.lang.annotation.*;

@Documented
@Constraint(validatedBy = LegalAgeValidator.class)
@Target({ElementType.FIELD})
@Retention(RetentionPolicy.RUNTIME)
public @interface LegalAge {

    String message() default "{legal.age}";

    Class<?>[] groups() default {};

    Class<? extends Payload>[] payload() default {};
}
