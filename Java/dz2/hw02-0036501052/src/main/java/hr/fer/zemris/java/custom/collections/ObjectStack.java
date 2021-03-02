package hr.fer.zemris.java.custom.collections;

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
		col.add(value);
	}
	
	/**
	 * Returns the peek value from stack and removes that value from stack
	 * @return peek value from stack
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

	@Override
	public boolean equals(Object arg0) {
		if((arg0 instanceof ObjectStack) == false) {
			return false;
		}
		
		ObjectStack other = (ObjectStack) arg0;
		
		if(size() != other.size()) {
			return false;
		}
		
		ObjectStack help1 = new ObjectStack();
		ObjectStack help2 = new ObjectStack();
		
		while((this.isEmpty() && other.isEmpty()) == false) {
			Object o1 = pop();
			Object o2 = other.pop();
			
			if(o1.equals(o2) == false) {
				move(help1,this);
				move(help2,other);
				
				return false;
			}
			
			help1.push(o1);
			help2.push(o2);
			
		}
		
		move(help1,this);
		move(help2,other);
		
		return true;
	}
	
	/**
	 * helper method for moving object from one stack to another
	 * @param source Stack which acts as a source of object
	 * @param target Stack which we want to fill
	 */
	private void move(ObjectStack source,ObjectStack target) {
		while(source.isEmpty() == false) {
			target.push(source.pop());
		}
	}
	
	
	
	
	
}
