package hr.fer.zemris.java.custom.scripting.nodes;

import java.util.Objects;

import hr.fer.zemris.java.custom.scripting.elems.Element;
import hr.fer.zemris.java.custom.scripting.elems.ElementFunction;

/**
 * @author Win10
 *
 */
public class EchoNode extends Node{
	private Element[] elements;

	public EchoNode(Element[] elements) {
		this.elements = Objects.requireNonNull(elements);
	}
	
	public Element[] getElements() {
		return elements;
	}

	@Override
	public String toString() {
		String str = "{$=";
		for(Element el:elements) {
			str += el.asText() + " "; 
		}
		return str + "$}";
	}
	
	
	
}
