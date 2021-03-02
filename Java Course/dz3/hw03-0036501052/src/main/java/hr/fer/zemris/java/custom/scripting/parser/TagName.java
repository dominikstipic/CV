package hr.fer.zemris.java.custom.scripting.parser;

/**
 * Enumeration which lists all possible tag names:
 * FOR,ECHO,END tags.
 * @author Dominik Stipic
 *
 */
public enum TagName {
	FOR("FOR"),
	ECHO("="),
	END("END");
	
	private String value;

	/**
	 * Constructor for giving enum string value
	 * @param value - desired string value which will represent enum  
	 */
	private TagName(String value) {
		this.value = value;
	}

	@Override
	public String toString() {
		return value;
	}
	
	
}
