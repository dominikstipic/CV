package hr.fer.zemris.java.custom.collections;

/**
 * Class which represents Linked list collections
 * @author Dominik Stipić
 * @version 1.0
 *
 */
public class LinkedListIndexedCollection extends Collection {
	private int size;
	private ListNode first;
	private ListNode last;

	/**
	 * Class which represents node of linked list
	 * Node contains reference on next and previous node.It also contains 
	 * one varibale for storing data
	 * @author Dominik Stipic
	 * @version 1.0
	 */
	public static class ListNode {
		ListNode next;
		ListNode previous;
		Object value;

		public ListNode(Object value, ListNode previous, ListNode next) {
			this.next = next;
			this.previous = previous;
			this.value = value;
		}

	}

	/**
	 * Default constructor for creating linked list
	 */
	public LinkedListIndexedCollection() {
		first = last = null;
		size = 0;
	}

	/**
	 * Constructor for creating linked list with the same elements as passed collection
	 * @param collection which elements will be inserted in created list
	 * @throws NullPointerException - if the given collection is null 
	 */
	public LinkedListIndexedCollection(Collection collection) {
		addAll(collection);
		size = collection.size();
	}

	////////////////////////////////////////////////////////// 

	@Override
	public void add(Object value) {
		if (value == null) {
			throw new IllegalArgumentException("Null values for this collection are prohibited");
		}

		if (isEmpty()) {
			ListNode newNode = new ListNode(value, null, null);
			first = newNode;
			last = newNode;
		} else {
			ListNode newNode = new ListNode(value, last, last.next);
			last.next = newNode;
			last = newNode;
		}

		++size;
	}

	@Override
	public int size() {
		return size;
	}

	@Override
	public void clear() {
		first = last = null;
		size = 0;
	}

	/**
	 * Retrives an element on given index
	 * @param index of element which is needed
	 * @return element at specified index 
	 * @throws IndexOutOfBoundsException - if the index is not at defined interval
	 */
	public Object get(int index) {
		if (!(index >= 0 && index <= size - 1)) {
			throw new IndexOutOfBoundsException("index isn't in defined interval");
		}
		ListNode currentNode;
		if (index == 0) {
			return first.value;
		} else if (index == size - 1) {
			return last.value;
		} else {
			currentNode = getPosition(index);
		}

		return currentNode.value;
	}

	/**
	 * Inserts given element at given positon 
	 * @param value which is going to be stored in collection
	 * @param position in which element is going to be stored
	 * @throws IndexOutOfBoundsException - if the index is not at defined interval
	 * @throws NullPointerException - if the value is null
	 */
	public void insert(Object value, int position) {
		if (!(position >= 0 && position <= size)) {
			throw new IndexOutOfBoundsException("This position doesn't exist in this list");
		}
		if(value == null) {
			throw new NullPointerException("value cannot be null");
		}

		if (isEmpty() || position == size) {
			add(value);
			return; // size is incremented in add method
		} else if (position == 0) {
			ListNode newNode = new ListNode(value, null, first);
			first.previous = newNode;
			first = newNode;
		} else {
			ListNode current = getPosition(position).previous;
			ListNode newNode = new ListNode(value, current, current.next);
			current.next.previous = newNode;
			current.next = newNode;

		}

		++size;

	}

	/**
	 * Removes an element at given index
	 * @param index of element which is going to be deleted form collection
	 * @throws IndexOutOfBoundsException - if the index is not at defined interval
	 */
	public void remove(int index) {
		if (!(index >= 0 && index <= size - 1)) {
			throw new IndexOutOfBoundsException("This position doesn't exist in this list");
		}

		if (index == 0) {
			first = first.next;
			first.previous.next = null;
		} else if (index == size - 1) {
			last = last.previous;
			last.next.previous = null;
			last.next = null;
		} else {
			ListNode current = getPosition(index).previous;
			ListNode delete = current.next;

			current.next = delete.next;
			delete.next.previous = current;

			delete.next = delete.previous = null;
		}

		--size;
	}

	@Override
	public boolean contains(Object value) {
		if (value == null) {
			return false;
		}
		if (isEmpty()) {
			return false;
		}

		for (ListNode current = first; current != null; current = current.next) {
			if (current.value.equals(value)) {
				return true;
			}
		}

		return false;
	}

	@Override
	public Object[] toArray() {
		if (isEmpty()) {
			return null;
		}

		Object[] arr = new Object[size];
		int index = 0;
		for (ListNode current = first; current != null; current = current.next) {
			arr[index] = current.value;
			++index;
		}

		return arr;
	}

	@Override
	public void forEach(Processor processor) {
		for (ListNode current = first; current != null; current = current.next) {
			processor.process(current.value);
		}
	}

	@Override
	public boolean remove(Object value) {
		if(value == null) {
			throw new NullPointerException("value cannot be null");
		}
		
		if(contains(value)) {
			int index = indexOf(value);
			remove(index);
			--size;
			
			return false;
		}
		else {
			return false;
		}
	}
	
	/**
	 * Method which gets the index of given element in collection
	 * @param value of element which index is wanted
	 * @return index of element
	 * @throws NullPointerException - if the value is null 
	 */
	public int indexOf(Object value) {
		if(isEmpty() && value == null) {
			return 0;
		}
		else if(contains(value)) {
			int index = 0;
			for(ListNode current = first; current != null; current = current.next) {
				if(current.value.equals(value)) break;
				++index;
			}
			return index;
		}
		else {
			return -1;
		}
	}
	
	/////////////////////////////////////////////////////////////////// 


	@Override
	public String toString() {
		if (isEmpty() == false) {
			String s = "[";
			for (ListNode current = first; current != null; current = current.next) {
				s += current.value + ",";
			}

			char c[] = s.toCharArray();
			c[c.length - 1] = ']';

			return String.valueOf(c);
		} else {
			return "empty";
		}

	}

	/**
	 * Helper method-> 
	 * Method which returns ListNode at given positon
	 * @param index of node which is needed
	 * @return ListNode which is at specified positon
	 */
	private ListNode getPosition(int index) {
		ListNode current;

		if (index > size / 2) {
			current = last;
			for (int counter = size - 1; counter != index; --counter) {
				current = current.previous;
			}
		} else {
			current = first;
			for (int counter = 0; counter != index; ++counter) {
				current = current.next;
			}
		}

		return current;
	}

	@Override
	public boolean equals(Object arg0) {
		if((arg0 instanceof LinkedListIndexedCollection) == false) {
			return false;
		}
		
		LinkedListIndexedCollection col = (LinkedListIndexedCollection) arg0;
		
		if(size != col.size) {
			return false;
		}
		
		for(int i = 0; i < size; ++i) {
			if(get(i).equals(col.get(i)) == false) {
				return false;
			}
		}
		
		return true;
	}
	
	

	
	
}
