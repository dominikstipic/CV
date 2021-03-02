package hr.fer.zemris.java.custom.scripting.parser;


/**
 * Exception which occurs when in process of parsing
 * @author Dominik Stipic
 *
 */
public class SmartScriptParserException extends RuntimeException{
	private static final long serialVersionUID = 1L;
	
	/**
	 * Default Constructor
	 */
	public SmartScriptParserException( ) {
	}
	/**
	 * Constructor with additional error message
	 * @param message which specifies more details about exception
	 */
	public SmartScriptParserException(String message) {
		super(message);
	}
	
}
