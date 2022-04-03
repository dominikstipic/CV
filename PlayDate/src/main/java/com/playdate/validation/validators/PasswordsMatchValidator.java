package com.playdate.validation.validators;

import com.playdate.forms.RegisterForm;
import com.playdate.validation.annotations.PasswordsMatch;
import lombok.SneakyThrows;
import org.apache.commons.beanutils.BeanUtils;

import javax.validation.ConstraintValidator;
import javax.validation.ConstraintValidatorContext;

public class PasswordsMatchValidator implements ConstraintValidator<PasswordsMatch, RegisterForm> {
   private String passwordFieldName;
   private String confirmPasswordFieldName;
   private String message;

   public void initialize(PasswordsMatch constraint) {
      passwordFieldName = constraint.first();
      confirmPasswordFieldName = constraint.second();
      message = constraint.message();
   }

   @SneakyThrows
   public boolean isValid(RegisterForm value, ConstraintValidatorContext context) {
      final String password = BeanUtils.getProperty(value, passwordFieldName);
      final String confirmPassword = BeanUtils.getProperty(value, confirmPasswordFieldName);

      boolean valid = password == null || password.isEmpty() || password.equals(confirmPassword);

      if (!valid) {
         context.buildConstraintViolationWithTemplate(message)
                 .addPropertyNode(passwordFieldName)
                 .addConstraintViolation()
                 .disableDefaultConstraintViolation();
      }

      return valid;
   }
}
