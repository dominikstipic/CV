package hr.fer.zemris.java.custom.scripting.nodes;

import java.util.Objects;

public class Node {
	private ArrayIndexedCollection col;
	
	public Node() {
		
	}

	public void addChildNode(Node child) {
		if(col == null) {
			col = new ArrayIndexedCollection();
		}
		col.add(child);
	}
	
	public int numberOfChildren() {
		if(col == null ) {
			return 0;
		}
		return col.size();
	}
	
	public Node getChild(int index) {
		Objects.requireNonNull(col);
		return (Node) col.get(index);
	}
	
}
