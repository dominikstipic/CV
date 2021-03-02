package hr.fer.zemris.java.hw03.prob1;

/**
 * Functional interface representing generic test
 * @author Dominik Stipic
 *
 */
@FunctionalInterface
public interface Tester {
	/**
	 * generic test
	 * @param value - integer which is used in test
	 * @return appropriate boolean value
	 */
	boolean test(int value);
}
