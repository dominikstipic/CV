package hr.fer.zemris.java.custom.scripting.elems;
import static java.lang.Character.isLetter;
/**
 * Element which stores string values of the expression
 * @author Dominik Stipic 
 *
 */
public class ElementString extends Element{
	private String value;

	/**
	 * Constructor for creating this object
	 * @param value- String value which is going to be stored
	 */
	public ElementString(String value) {
		this.value = value;
	}

	@Override
	public String asText() {
		char arr[] = value.toCharArray();
		StringBuilder builder = new StringBuilder();
		
		for(int i = 0; i < arr.length; ++i) {
			if(arr[i] == '"') {
				builder.append('\\');
				}
			builder.append(arr[i]);
		}
		return "\""+builder.toString()+"\"";
	}

	
	
	
}
