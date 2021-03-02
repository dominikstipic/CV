package hr.fer.zemris.java.custom.scripting.elems;

/**
 * Element which stores representation of variable of given expression
 * @author Dominik Stipic
 *
 */
public class ElementVariable extends Element{
	private String name;

	/**
	 * Constructor for creating this object
	 * @param name - name of variable
	 */
	public ElementVariable(String name) {
		this.name = name;
	}
	
	@Override
	public String asText() {
		return name;
	}
	
	
}
