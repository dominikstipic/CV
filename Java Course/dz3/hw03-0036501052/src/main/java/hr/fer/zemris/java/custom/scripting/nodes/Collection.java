package hr.fer.zemris.java.custom.scripting.nodes;

/**
 * Represents general collections for objects.
 * Its purpose is to store objects 
 * @author Dominik Stipić
 * @version 1.0
 */
public class Collection { 
	
	/**
	 * Default constructor
	 */
	protected Collection() {
		
	}
	
	/**
	 * Checks if the collection is empty
	 * @return true - if collection is empty
	 * 		   false - collectio is not empty
	 */
	public boolean isEmpty() {
		return size() == 0 ? true : false;
	}
	
	/**
	 * Cheks the number of elements in collection
	 * @return the number of currently stored objects in this collection
	 */
	public int size() {
		return 0;
	}
	
	/**
	 * Adds the given object into this collection
	 * @param value object which will be added
	 * @throws NullPointerException - if the specified element is null 
	 */
	public void add(Object value) {
		
		
	}
	
	/**
	 * Checks if the specified object exist in this collection
	 * @param value element whose presence we want to test
	 * @return true - if element is in collection
	 * 		   false - if element isn't in collection
	 */
	boolean contains(Object value) {
		return false;
	}
	
	/**
	 * Removes specified object from the collection,if object exist in collection
	 * @param value which we want to remove from collection
	 * @return true - if object is removed from collection
	 * 		   false - if object doesn't exist in collection	
	 */
	public boolean remove (Object value) {
		return false;
	}
	
	/**
	 * Returns an array containing all of the elements in this collection
	 * @return an array containing all of the elements
	 * @throws UnsupportedOperationException - if the operation is not supported by this collection
	 */
	public Object[] toArray() {
		throw new UnsupportedOperationException("Method isn't supported");
	}
	
	/**
	 * Method calls process method from Processor class for each element of collection
	 * @param processor which process method will be called.
	 * process method is defined by user
	 * @throw NullPointerException - if the processor is null
	 */
	public void forEach(Processor processor) {
		
	}
	
	/**
	 * Method adds into this collection all elements from the given collection.
	 * Other collection remains unchanged 
	 * @param other collection containing elements which we want add in this collection. 
	 * @throw NullPointerException - if the given collection is null
	 */
	public void addAll(Collection other) {
		if(other == null) {
			throw new NullPointerException("Given collection musn't be null");
		}
		
		class ProccesorLocaly extends Processor{
			@Override
			public void process(Object value) {
				add(value);
			}
		}
		
		ProccesorLocaly processor = new ProccesorLocaly();
		other.forEach(processor);
	}
	
	/**
	 * Removes all elements from this collection
	 */
	public void clear() {
		
	}
}
