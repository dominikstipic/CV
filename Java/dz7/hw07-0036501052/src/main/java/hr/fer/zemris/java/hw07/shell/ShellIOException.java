package hr.fer.zemris.java.hw07.shell;

/**
 * Exception is thrown when peoblem ocurrs in dealing with <code>MyShell</code> 
 * @author Dominik Stipić
 */
public class ShellIOException extends RuntimeException {
	private static final long serialVersionUID = 1L;
	
	/**
	 * Creates new ShellIOException with appropriate message
	 * @param message which will be printed to user
	 */
	public ShellIOException(String message) {
		super(message);
	}
}
