package hr.fer.zemris.java.hw03.prob1;

/**
 * Exception which represents mistake which has happened during lexical analysis of
 * given text
 * @author Dominik Stipic
 *
 */
public class LexerException extends RuntimeException {
	private static final long serialVersionUID = 1L;
	
	/**
	 * Default constructor 
	 */
	public LexerException() {

	}

	/**
	 * Constructor with additional error message
	 * @param message which specifies more details about exception
	 */
	public LexerException(String message) {
		super(message);
	}
}
