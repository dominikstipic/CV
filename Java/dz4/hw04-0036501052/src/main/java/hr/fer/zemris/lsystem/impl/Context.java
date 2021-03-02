package hr.fer.zemris.lsystem.impl;

import java.util.Objects;

import hr.fer.zemris.java.custom.collections.EmptyStackException;
import hr.fer.zemris.java.custom.collections.ObjectStack;

/**
 * Stores turtle's state by using stack data structure.
 * This feature provides decision making and bactracking
 * @author Win10
 *
 */
public class Context {
	/**
	 * Stack in which turtle's state is saved 
	 */
	private ObjectStack stack;
	
	
	
	/**
	 *Default Constructor 
	 */
	public Context() {
		stack = new ObjectStack();
	}

	/**
	 * Gets current state of turtle 
	 * @return turtle's current state
	 */
	public TurtleState getCurrentState() {
		if(stack.isEmpty()) {
			throw new EmptyStackException("Cannot deliver current positon - stack is empty");
		}
		return (TurtleState) stack.peek();
	}
	
	/**
	 * pushes new turtle state in stack
	 * @param state - turtle state
	 * @throws NullPointerException - if the state is null
	 */
	public void pushState(TurtleState state) {
		Objects.requireNonNull(state);
		stack.push(state);
	}
	
	/**
	 * Pops state from stack.
	 * @throws EmptyStackException - if stack is empty
	 */
	public void popState() {
		if(stack.isEmpty()) {
			throw new EmptyStackException("stack is empty");
		}
		stack.pop();
	}
	
	/**
	 * Deletes all content from stack
	 */
	public void clearContext() {
		stack.clear();
	}
}
