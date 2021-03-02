package hr.fer.zemris.java.custom.scripting.elems;

/**
 * Element holding double value
 * @author Dominik Stipic
 *
 */
public class ElementConstantDouble extends Element{
	private double value;

	/**
	 * Constructor for creating element with double value 
	 * @param value which is going to be stored
	 */
	public ElementConstantDouble(double value) {
		this.value = value;
	}
	
	@Override
	public String asText() {
		return String.valueOf(value);
	}
}
