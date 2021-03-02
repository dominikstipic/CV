package hr.fer.zemris.java.custom.scripting.nodes;

/**
 * Represents textual part of source program
 * @author Dominik Stipic
 *
 */
public class TextNode extends Node{
	private String text;

	/**
	 * Creates textual node
	 * @param text 
	 */
	public TextNode(String text) {
		this.text = text;
	}

	/**
	 * Getter for atribute text 
	 * @return textual content of source program
	 */
	public String getText() {
		return text;
	}
	
	
	
}
