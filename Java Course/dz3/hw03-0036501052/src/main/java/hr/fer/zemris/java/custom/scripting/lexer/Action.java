package hr.fer.zemris.java.custom.scripting.lexer;

/**
 * Functional interface which represents some generic action.
 * @author Dominik Stipic
 *
 */
@FunctionalInterface
public interface Action {
	/**
	 * Method which represents some generic action
	 * @param word - String which is used in action
	 */
	void action(String word);
}
