package hr.fer.zemris.java.gui.layouts;

/**
 * Exceeption thrown by {@link Calculator}
 * @author Dominik Stipić
 *
 */
public class CalcLayoutException extends RuntimeException{
	private static final long serialVersionUID = 1L;

	/**
	 * Creates CalcLayoutException with provided error message
	 * @param message
	 */
	public CalcLayoutException(String message) {
		super(message);
	}
}
