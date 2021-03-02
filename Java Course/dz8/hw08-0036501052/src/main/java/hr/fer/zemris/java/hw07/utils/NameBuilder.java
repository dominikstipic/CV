package hr.fer.zemris.java.hw07.utils;

/**
 * Generates parts of inputed expression by writing into StringBuilder, which is
 * provided as an argument in execute method.
 * 
 * @author Dominik Stipić
 *
 */
public interface NameBuilder {
	/**
	 * Writes content into info's StringBuilder
	 * @param info provided object which holds information about inputed text
	 */
	void execute(NameBuilderInfo info);
}
