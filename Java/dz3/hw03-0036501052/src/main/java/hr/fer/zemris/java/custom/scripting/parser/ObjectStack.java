package hr.fer.zemris.java.custom.scripting.parser;


import java.util.Objects;

import hr.fer.zemris.java.custom.scripting.nodes.ArrayIndexedCollection;


/**
 * Class which represents stack with its common methods.
 * This is a stack for holding generic object data types 
 * @author Dominik Stipic
 *@version 1.0
 */
public class ObjectStack {
	private ArrayIndexedCollection col;
	
	/**
	 *Creates new ObjectStack 
	 */
	public ObjectStack() {
		col = new ArrayIndexedCollection();
	}
	
	/**
	 * Checks if stack is empty
	 * @return true - if stack is empty 
	 * 			false - otherwise
	 */
	public boolean isEmpty() {
		return col.isEmpty();
	}
	
	/**
	 * @return number of objects in stack
	 */
	public int size() {
		return col.size();
	}
	
	/**
	 * sets given object on stack
	 * @param value which is going to be pushed on stack
	 */
	public void push(Object value) {
		Objects.requireNonNull(value);
		col.add(value);
	}
	
	/**
	 * Returns the peek value from stack and removes that value from stack
	 * @return peek value from stack
	 * @throws EmptyStackException if stack is empty
	 */
	public Object pop() {
		if(col.isEmpty()) {
			throw new EmptyStackException("Stack is empty");
		}
		else {
			Object value = col.get(col.size()-1);
			col.remove(col.size()-1);
			return value;
		}
	}
	
	/**
	 * Returns peek value of this stack without removing it
	 * @return peek value of stack
	 * @throws EmptyStackException - if the stack is empty
	 */
	public Object peek() {
		if(col.isEmpty()) {
			throw new EmptyStackException("Stack is empty");
		}
		else {
			return col.get(col.size()-1);
		}
	}
	
	/**
	 * Erases all content from this stack
	 */
	public void clear() {
		col.clear();
	}

	@Override
	public String toString() {
		return col.toString();
	}
}
