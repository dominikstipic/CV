package hr.fer.zemris.java.custom.scripting.elems;

/**
 * Element representing mathematical operation
 * @author Dominik Stipic 
 *
 */
public class ElementOperator extends Element{
	private String symbol;

	/**
	 * Constructor for creating this object
	 * @param symbol String which represents math operation
	 */
	public ElementOperator(String symbol) {
		this.symbol = symbol;
	}

	@Override
	public String asText() {
		return symbol;
	}
	
	
}
