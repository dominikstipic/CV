package hr.fer.zemris.java.custom.scripting.elems;

/**
 * Element representing function
 * @author Dominik Stipic
 *
 */
public class ElementFunction extends Element{
	private String name;

	/**
	 * Constructor for creating function element 
	 * @param name of function 
	 */
	public ElementFunction(String name) {
		this.name = name;
	}

	@Override
	public String asText() {
		return "@"+name;
	}

	

	
	
	
	
}
