package hr.fer.rasus.client.exceptions;

public class RegistrationError extends Exception{
	private static final long serialVersionUID = 1L;

	public RegistrationError(String errorMessage) {
        super(errorMessage);
    }
}
