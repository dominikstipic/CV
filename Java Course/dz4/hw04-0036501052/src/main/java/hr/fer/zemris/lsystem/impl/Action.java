package hr.fer.zemris.lsystem.impl;

/**
 * Functional interface which represents some generic action.
 * @author Dominik Stipic
 *
 */
@FunctionalInterface
public interface Action {
	/**
	 * Method which represents some generic action
	 * @param tokens 
	 * @param size
	 */
	void action(String[] tokens, int size);
}
