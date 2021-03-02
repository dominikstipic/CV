package hr.fer.zemris.java.custom.scripting;

import java.util.EmptyStackException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Objects;

/**
 * Data Structure which connects key with corresponding stack.
 * Every stack has unique key which is String.
 * Stack is modeled with <code>MultistackEntry</code>
 * which stores <code>ValueWrapper</code>
 * @author Dominik Stipic 
 *
 */
public class ObjectMultistack {
	/**
	 * Map which connects the String to corresponding <code>MultistackEntry</code> 
	 */
	private Map <String,MultistackEntry> stackMap = new HashMap<>();
	
	/**
	 * Pushes <code>ValueWrapper</code> value on stack with corresponding key
	 * @param name key of stack
	 * @param valueWrapper object which is going to be pushed
	 */
	public void push (String name, ValueWrapper valueWrapper) {
		Objects.requireNonNull(name);
		if(!stackMap.containsKey(name)) {
			stackMap.put(name, new MultistackEntry());
		}
		
		stackMap.get(name).pushValue(valueWrapper);
	}
	
	/**
	 * Pops <code>ValueWrapper</code> value from stack with corresponding key
	 * @param name key of stack
	 * @return valueWrapper object which is going to be pushed
	 * @throws EmptyStackException - if stack is empty
	 */
	public ValueWrapper pop(String name) {
		Objects.requireNonNull(name);
		if(!stackMap.containsKey(name)) {
			throw new IllegalArgumentException("Provided key without corresponding stack");
		}
		return stackMap.get(name).popValue();
	}
	
	/**
	 * Returns <code>ValueWrapper </code> object from the top of corresponding stack 
	 * @param name Stack's key
	 * @return <code>ValueWrapper </code> object from the top of stack
	 * @throws EmptyStackException - if stack is empty
	 */
	public ValueWrapper peek (String name){
		Objects.requireNonNull(name);
		if(!stackMap.containsKey(name)) {
			throw new IllegalArgumentException("Provided key without corresponding stack");
		}
		return stackMap.get(name).getPeek(); 
	}
	
	/**
	 * Checks if the stack is empty
	 * @param name Stack's key
	 * @return true - if stack is empty 
	 * false . otherwise
	 */
	public boolean isEmpty(String name) {
		Objects.requireNonNull(name);
		return stackMap.get(name).isStackEmpty();
	}
	
	/**
	 * Models the stack with the standard Stack API 
	 * @author Dominik Stipic
	 *
	 */
	private static class MultistackEntry{
		/**
		 * Stack which holds  <code>ValueWrapper </code>  objects
		 */
		private LinkedList<ValueWrapper> stack  = new LinkedList<>();

		/**
		 * Pushes <code>ValueWrapper </code> object on top of stack
		 * @param value object to be pushed
		 */
		public void pushValue(ValueWrapper value) {
			stack.add(value);
		}
		
		/**
		 * Pops the <code>ValueWrapper </code> from the stack and removes it 
		 * @return <code>ValueWrapper </code> object from top of stack
		 */
		public ValueWrapper popValue() {
			if(stack.isEmpty()) {
				throw new EmptyStackException();
			}
			ValueWrapper value = stack.get(stack.size()-1);
			stack.remove(stack.size()-1);
			return value;
		}
		
		/**
		 * Retrurns <code>ValueWrapper </code> object form the top of the
		 * stack without removing it from the stack
		 * @return <code>ValueWrapper </code> object from the top of stack
		 */
		public ValueWrapper getPeek() {
			if(stack.isEmpty()) {
				throw new EmptyStackException();
			}
			return stack.get(stack.size()-1);
		}
		
		/**
		 * Checks if the stack is empty 
		 * @return corresponding boolean value
		 */
		public boolean isStackEmpty() {
			return stack.size() == 0;
		}
		
		
	}
	
}
