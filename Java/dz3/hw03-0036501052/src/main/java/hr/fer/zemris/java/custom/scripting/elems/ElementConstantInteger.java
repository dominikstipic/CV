package hr.fer.zemris.java.custom.scripting.elems;

/**
 *Element holding double value
 * @author Dominik Stipic
 *
 */
public class ElementConstantInteger extends Element{
	private int value;
	
	/**
	 * Constructor for creating element with int value 
	 * @param value which is going to be stored
	 */
	public ElementConstantInteger(int value) {
		this.value = value;
	}
	
	@Override
	public String asText() {
		return String.valueOf(value);
	}
}
