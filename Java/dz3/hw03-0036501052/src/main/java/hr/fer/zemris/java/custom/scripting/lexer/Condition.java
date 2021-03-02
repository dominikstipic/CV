package hr.fer.zemris.java.custom.scripting.lexer;

/**
 *Functional interface representing generic conditon 
 * @author Dominik Stipic
 */
@FunctionalInterface
public interface Condition {
	/**
	 * generic codition which is tested
	 * @return appropriate boolean value
	 */
	boolean condition();
}
