package hr.fer.zemris.java.custom.collections;

import java.util.Objects;

/**
 * Collection which represents mapping.
 * It hold entities know by TableEntry which represents 
 * two connected pairs (key,value)
 * key musnt be null, while value can be any data 
 * @author Dominik Stipic
 * @version1.0
 */
public class Dictionary {
	/**
	 * Array for storing TableEntries
	 */
	private ArrayIndexedCollection table;
	
	/**
	 * Default Construnctor
	 */
	public Dictionary() {
		table = new ArrayIndexedCollection();
	}

	/**
	 * @return true - if collectio is empty
	 * false - otherwise
	 */
	public boolean isEmpty() {
		return table.isEmpty();
	}

	/**
	 * Returns number of stored TableEntries
	 * @return Size of this Dictionary
	 */
	public int size() {
		return table.size();
	}

	/**
	 * Deletes content of this dicitonary 
	 */
	public void clear() {
		table.clear();
	}
	
	/**
	 * Stores given pair (key,value) in this dictionary.
	 * Key musnt be null value while value can be anything
	 * If this dictionary already stores TableEntry with same key value as 
	 * given, the old TableEntry will be replaced with newer given version
	 * @param key Identificator for TableEntry 
	 * @param value	Which is going to be stored
	 * @throws NullPointerException - if key is null
	 */
	public void put(Object key, Object value) {
		Objects.requireNonNull(key);
		if(table.contains(new TableEntry(key,null))) {
			table.remove(new TableEntry(key, null));
			table.add(new TableEntry(key, value));
		}
		else {
			table.add(new TableEntry(key, value));
		}
		
	}
	
	/**
	 * Retrives value from dictionary which fits the given key.
	 * If dictionary doesnt hold given key null is returned
	 * @param key - identificator for wanted value
	 * @return value which is connected to given key
	 */
	public Object get (Object key) {
		if(!table.contains(new TableEntry(key, null))) {
			return null;
		}
		
		TableEntry entry = null;
		for(int i = 0;i < table.size(); ++i) {
			entry = (TableEntry) table.get(i);
			if(entry.equals(new TableEntry(key, null))) {
				break;
			}
		}
		
		return entry.value;
	}

	/**
	 * The encapsulated data which is actually stored in this dicitionary.
	 * It contains unique key and appropriate value
	 * @author Dominik Stipic
	 *
	 */
	private class TableEntry{
		private Object key;
		private Object value;
		
		/**
		 * Constructor for creating table entry
		 * @param key unique identifier of this table entry 
		 * @param value - value which is connected to key.
		 */
		public TableEntry(Object key, Object value) {
			this.key = key;
			this.value = value;
		}

		@Override
		public boolean equals(Object arg0) {
			if(arg0 instanceof TableEntry == false)
				return false;
			
			TableEntry other = (TableEntry) arg0;
			
			return key.equals(other.key);
		}
		
		
	}
	
}
