package hr.fer.zemris.java.hw07.utils;

/**
 * Stores StringBuilder with whome the process 
 * of generating text starts.  
 * @author Dominik Stipić
 *
 */
public interface NameBuilderInfo {
	/**
	 * gets stored String Builder
	 * @return StringBuilder
	 */
	StringBuilder getStringBuilder();

	/**
	 * Return the String represenation of captured sequnece
	 * @param index of group
	 * @return String representation of  captured data
	 */
	String getGroup(int index);
}
