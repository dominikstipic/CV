package hr.fer.zemris.java.custom.collections;

/**
 * Exception which occurs when dealing with Stack
 * @author Dominik Stipic
 *
 */
public class EmptyStackException extends RuntimeException{
	
	private static final long serialVersionUID = 1L;
	
	/**
	 * Default constructor
	 */
	public EmptyStackException () {
	}
	
	/**
	 * Constructor with additional error message
	 * @param message which specifies more details about exception
	 */
	public EmptyStackException (String message) {
		super(message);
	}
	

}
