package hr.fer.zemris.java.custom.collections;


import java.util.Arrays;
import java.util.Objects;

/**
 * Class which implement collection of resizable arrays. 
 * Duplicate elements are allowed but storage of null references is not allowed
 * 
 * @author Dominik Stipic
 *
 */
public class ArrayIndexedCollection extends Collection {
	private int size;
	private int capacity;
	private Object[] elements;
	public static final int DEFAULT_CAPACITY = 16;

	/**
	 * Creates new ArrayIndexedCollection with capacity of 16
	 */
	public ArrayIndexedCollection() {
		this(null, 16);
	}

	/**
	 * Creates new collection with capacity specified with initialCapacity
	 * @param initialCapacity initial capacity of collection
	 */
	public ArrayIndexedCollection(int initialCapacity) {
		this(null, initialCapacity);
	}

	/**
	 * Creates collection with the same elements as given collection
	 * @param other Collection which elements will be copied in this collection
	 * @throws NullPointerException - if passed argument is null
	 */
	public ArrayIndexedCollection(Collection other) {
		this(other, 16);
	}

	/**
	 * Creates new collection with the same elements as given collection and with specified initial capacity
	 * @param other Collection which elements will be copied in this collection
	 * @param initialCapacity initial capacity of this collection
	 */
	public ArrayIndexedCollection(Collection other, int initialCapacity) { 
		this.capacity = initialCapacity;
		elements = new Object[initialCapacity];
		if (other != null) {
			this.addAll(other);
		}

	}

	/////////////////////////////////////////////

	/**
	 * Inserts the given value in this collection at given positon.
	 * Method does not overwrite stored elements in this collection
	 * avarage complexity of this method is O(n)
	 * @param value element who will be inserted in this array collection
	 * @param position where element will be inserted
	 * @throws NullPointerException - when passed value is null
	 * @throws IndexOutOfBoundsException - when passed position isn't defined for this collection
	 */
	public void insert(Object value, int position) {
		if(value == null) {
			throw new NullPointerException("Null pointer cannot be inserted in this collection");
		}
		if (position >= 0 && position <= size) {
			
			if (needToReallocate() == true) {
				elements = reallocateArray(2 * capacity);
				capacity *= 2;
			}

			shiftArrayForOnePlace(position);
			elements[position] = value;
			++size;

		} else {
			throw new IndexOutOfBoundsException("This position isn't defined for this collection");
		}
	}

	/**
	 * Returns index of given value
	 * @param value value whose index is needed
	 * @return -1 -> if the value isn't stored in this collection or the null value is passed.
	 * 			Returns index of value otherwise.
	 */
	public int indexOf(Object value) {
		if (value == null) {
			return -1;
		}

		for (int i = 0; i < size; ++i) {
			if (elements[i].equals(value))
				return i;
		}
		
		return -1;
	}
	

	@Override
	public void add(Object value) {
		if (value == null) {
			throw new NullPointerException("You can not add null to this collection");
		}

		if (needToReallocate() == true) {
			elements = reallocateArray(2 * capacity);
			capacity *= 2;
		}

		elements[size] = value;
		++size;
	}

	@Override
	public void clear() {
		for (int i = 0; i < elements.length; ++i) {
			elements[i] = null;
		}

		size = 0;
	}
	
	@Override
	public boolean remove(Object value) {
		if(value == null) {
			throw new NullPointerException("value cannot be null");
		}
		if(contains(value)==true) {
			int index = indexOf(value);
			elements[index] = null;
			removeNullFromArray(index);
			--size;
			
			return true;
		}
		else {
			return false;
		}
	}

	/**
	 * Removes element on given index
	 * @param index of element which is going to be deleted
	 * @throws IndexOutOfBoundsException - if the index isn't in defined interval
	 */
	public void remove(int index) {
		if (!(index >= 0 && index <= size-1)) {
			throw new IndexOutOfBoundsException("index isn't in defined interval for this array");
		}

		elements[index] = null;
		if(index != size-1) {
			removeNullFromArray(index);
		}
		--size;
	}

	/**
	 * Returns the element from specified index
	 * @param index of element which is needed
	 * @return value of needed element
	 * @throws IndexOutOfBoundsException - if the index isn't in defined interval
	 */
	public Object get(int index) {
		if (!(index >= 0 && index <= size-1)) {
			throw new IndexOutOfBoundsException("index isn't in defined interval for this array");
		}

		return elements[index];
	}

	
	@Override
	public void forEach(Processor processor) {
		for (int i = 0; i < size; ++i) {
			processor.process(get(i));
		}
	}
	
	@Override
	public boolean contains(Object value) {
		if (value == null) {
			return false;
		}
		if (indexOf(value) == -1) {
			return false;
		} else {
			return true;
		}
	}

	@Override
	public Object[] toArray() {
		return elements;
	}

	@Override
	public int size() {
		return size;
	}

	/**
	 * Helper method for insertion ->
	 * Shifts array for one place from the given index
	 * Attention - when using this method for inserting elements in collection.It is neccesary 
	 * first to call this method and then increment size of collection 
	 * @param position index from which all the elements will be shifted by one place
	 */
	private void shiftArrayForOnePlace(int position) {
		for (int i = size - 1; i >= position; --i) {
			elements[i + 1] = elements[i];
		}

	}

	/**
	 * Helper method ->
	 * Reallocates new array with same elements but with different length
	 * @param newLength new array length 
	 * @return new updated array
	 */
	private Object[] reallocateArray(int newLength) {
		return Arrays.copyOf(elements, newLength);
	}

	/**
	 * Helper method -> 
	 * Checks if the array reallocation is needed
	 * @return appropriate boolean value
	 */
	private boolean needToReallocate() {
		return size == capacity ? true : false;
	}

	/**
	 * Helper method - when removing element from collection.
	 * Removes null value from collection.Removes 'holes' from collection array 
	 * Attention - when using this method for removing elements it is neccesary first to call this method and then decrement size
	 * of collection 
	 * @param index of element which is null and in collection
	 */
	private void removeNullFromArray(int index) {
		for (int i = index; i < size; ++i) {
			elements[i] = elements[i + 1];
		}
	}

	@Override
	public String toString() {
		String s = "[";

		for (Object o : elements) {
			if (o == null)
				break;
			s += o + ",";
		}

		char c[] = s.toCharArray();
		c[c.length - 1] = ']';

		return String.valueOf(c);
	}

	/**
	 * @return Current capacity of this collection
	 */
	public int getCapacity() {
		return capacity;
	}
	

	@Override
	public boolean equals(Object arg0) {
		if((arg0 instanceof ArrayIndexedCollection) == false) {
			return false;
		}
		
		ArrayIndexedCollection col = (ArrayIndexedCollection) arg0;
		
		if(size != col.size) {
			return false;
		}
		
		for(int i = 0; i < size ;++i ) {
			if(elements[i].equals(col.elements[i]) == false) {
				return false;
			}  
		}
		
		return true;
	}

	
	
}
