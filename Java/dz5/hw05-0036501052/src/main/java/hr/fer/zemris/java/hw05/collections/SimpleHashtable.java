package hr.fer.zemris.java.hw05.collections;

import static java.lang.Math.abs;
import static java.lang.Math.ceil;
import static java.lang.Math.log;
import static java.lang.Math.pow;

import java.util.ConcurrentModificationException;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Objects;


/**
 * Parametrizirana kolekcija koj predstavlja tablicu rasprsenog adresiranja,
 * Objekti koji se spremaju u ovu kolekciju se par - kljuc vrijednost gdje se
 * vrijednost dohvaca uz pomoc kljuca.
 * Par kluc-vrijednost modeliran je s <code>TableEntry</code>
 * <code>key</code> ne moze biti null, dok <code>vrijednost</code> moze biti bilo sto.
 * Preljevna politika je implementirana pomocu vezane liste   
 * @author Dominik Stipic
 * @param <K> Tip podataka od kljuca
 * @param <V>Tip podatka od vrijednosti
 */
public class SimpleHashtable <K,V> implements Iterable<SimpleHashtable.TableEntry<K, V>>{
	/**
	 * Trenutni kapacitet tablice
	 */
	private int tableCapacity;
	/**
	 * Interno polje u koje se spremaju parovi kljuc-vrijednost
	 */
	private TableEntry<K,V> table [];
	/**
	 * broj spremljenih objekata
	 */
	private int numOfEntries = 0;
	/**
	 * broj akcija koje su promjenile strukturu tablice
	 */
	private int modificationCount = 0;
	/**
	 * zastavica koja govori dali je u tijeku realokacija tablice radi efikasnosti 
	 */
	private boolean efficiencyWorkRunning = false;
	
	/**
	 * ako je tablica 75% popunjena od max kapaciteta -> realokacija
	 */
	private final double EFFICIENCY_THRESHOLD = 0.75;
	/**
	 * pretpostavljena velicina tablice
	 */
	private static final int DEFAULT_SIZE = 16;
	/**
	 * faktor povecanja tablice kod realokacije
	 */
	private final int REALLOC_FACTOR = 2;
	
	
	
	/**
	 * pretpostavljeni konstruktor
	 */
	public SimpleHashtable() {
		this(DEFAULT_SIZE);
	}
	
	/**
	 * Konstruktor koji generira tablicu zadane velicine
	 * @param size velicina nove tablice
	 * @throws IllegalArgumentException ako je velicna tablice manja od 1
	 */
	@SuppressWarnings("unchecked")
	public SimpleHashtable(int size) {
		if(size < 1 ) {
			throw new IllegalArgumentException("Table size must be bigger then 1");
		}
		int x = (int)ceil(log(size)/log(2));
		this.tableCapacity = (int)pow(2, x);
		
		table = (TableEntry<K,V> []) new TableEntry[tableCapacity];
	}
	
	/**
	 * Umece novi <code>TableEntry</code>  kljuc-vrijednost u tablicu.
	 * Ako vec postoji kljuc u tablici, on se zamjenjuje sa 
	 * danom vrijednoscu.
	 * @param key kljuc objekta
	 * @param value vrijednst uz dani kljuc
	 * @throws NullPointerException - ako je <code>key</code> null
	 */
	public void put(K key, V value) {
		Objects.requireNonNull(key, "Key cannot be null");
		efficiencyControl();
		
		TableEntry<K,V> newEntry = new TableEntry<>(key, value, null);
		int index = calculateHash(key);
		TableEntry<K, V> entry = table[index];
		
		if(entry == null) {
			table[index] = newEntry;
			++numOfEntries;
			if(!efficiencyWorkRunning)++modificationCount;
			return;
		}
		
		while(true) {
			if(key.equals(entry.key)) {
				entry.setValue(value);
				return;
			}
			if(entry.next == null)break;
			entry = entry.next;
		}
		
		++numOfEntries;
		if(!efficiencyWorkRunning)++modificationCount;
		entry.next = newEntry;
	}
	
	/**
	 * Obavlja posao realokacije tablice samo ako je tablica 
	 * 75 % popunjenja od maksimalnog kapaciteta.
	 * Nova tablica ima dvostruko veci kapacitet
	 */
	@SuppressWarnings("unchecked")
	private void efficiencyControl() {
		if(numOfEntries >= tableCapacity*EFFICIENCY_THRESHOLD) {
			efficiencyWorkRunning = true;
			TableEntry<K, V> copies[] = (TableEntry<K, V> []) new TableEntry[numOfEntries];
			int index = 0;
			
			for(int i = 0; i < tableCapacity; ++i) {
				if(table[i] != null) {
					TableEntry<K, V> entry = table[i];
					while(entry != null) {
						copies[index] = entry;
						++index;
						entry = entry.next;
					}
				}
			}
			tableCapacity *= REALLOC_FACTOR;
			numOfEntries = 0;
			table = (TableEntry<K,V> []) new TableEntry[tableCapacity];
			
			for(int i = 0; i < copies.length; ++i) {
				K key = copies[i].getKey();
				V value = copies[i].getValue();
				put(key, value);
			}
			efficiencyWorkRunning=false;
		}
	}
	
	/**
	 * Brise sve <code>TableEntry</code> iz tablice
	 */
	@SuppressWarnings("unchecked")
	public void clear() {
		++modificationCount;
		numOfEntries = 0;
		table = (TableEntry<K,V> []) new TableEntry[table.length];	//set all to null
	}
	
	/**
	 * Dohvaca korespodentnu vrijednost za dani kljuc.
	 * @param key
	 * @return korespodentna vrijednost za dani kljuc.
	 * 		   Ako kljuc ne postoji u tablici,vraca se null
	 */
	public V get(K key) {
		if(key == null) {
			return null;
		}
		else if(!containsKey(key)) {
			return null;
		}
		else {
			int index = calculateHash(key);
			TableEntry<K, V> entry = table[index];
			
			while(entry != null) {
				if(entry.getKey().equals(key)) {
					return entry.getValue();
				}
				entry = entry.next;
			}
			
			return null;
		}
	}
	
	/**
	 * Pretrazuje tablicu u O(1) slozenosti i daje informaciju 
	 * dali postoji <code>TableEntry</code> s danim kljucem
	 * @param key kljuc pomocu kojeg se pretrazuje tablica
	 * @return true - postoji takav <code>TableEntry</code>,false inace
	 * @throws NullPointerException - ako je <code>key</code> null
	 * 
	 */
	public boolean containsKey(K key) {
		Objects.requireNonNull(key);
		TableEntry<K,V> entry = table[calculateHash(key)];
		
		while(entry != null) {
			if(entry.getKey().equals(key)) {
				return true;
			}
			entry = entry.next;
		}
		
		return false;
	}

	/**
	 * Pretrazuje tablicu u O(n) slozenosti i daje informaciju 
	 * dali postoji <code>TableEntry</code> s danom vrijednoscu
	 * @param value pomocu koje se pretrazuje tablica
	 * @return true - postoji takav <code>TableEntry</code>,false inace
	 */
	public boolean containsValue(V value) {
		for(int i = 0; i < tableCapacity; ++i) {
			if(table[i] != null) {
				TableEntry<K, V> entry = table[i];
				while(entry != null) {
					V checkedValue = entry.getValue();
					if(value == null) {
						if(checkedValue == null) {
							return true;
						}
					}
					else if(checkedValue != null && checkedValue.equals(value)) {
						return true;
					}
					entry = entry.next;
				}
			}
		}
		return false;
	}
	
	/**
	 * @return broj <code>TableEntry</code> u tablici
	 */
	public int size() {
		return numOfEntries;
	}

	/**
	 * Daje informaciju dalli je tablica prazna
	 * @return true-prazna,false-nije prazna
	 */
	public boolean isEmpty() {
		return numOfEntries == 0 ? true : false;
	}
	
	/**
	 * Unistava <code>TableEntry</code> s danim kljucem
	 * @param key kljuc para koji se zeli unistiti
	 */
	@SuppressWarnings("unused")
	public void remove(K key) {
		if(key == null ) {
			return;
		} 
		if(!containsKey(key)) {
			return;
		}
		--numOfEntries;
		++modificationCount;
		int index = calculateHash(key);

		TableEntry<K,V> entry = table[index];
		
		if(entry.getKey().equals(key)) {
			TableEntry<K,V> delete = entry;
			table[index] = entry.next;
			delete = null;
			return;
		}
		
		while(entry != null) {
			if(entry.next.getKey().equals(key)) {
				TableEntry<K,V> delete = entry.next;
				entry.next = delete.next;
				delete.next = null;
				return;
			}
			entry = entry.next;
		}
		
	}
	
	/**
	 * Racuna hash vrijednost pomocu kljuca 
	 * @param key kljuc pomocu kojeg se racuna hash vrijednost 
	 * @return hash vrijednost
	 */
	private int calculateHash(K key) {
		return abs(key.hashCode()) % tableCapacity;
	}
	
	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("[");
		for(int i = 0; i < tableCapacity; ++i) {
		if(table[i] != null) {
			TableEntry<K, V> entry = table[i];
			while(entry != null) {
				builder.append(entry.toString() + ",");
				entry = entry.next;
			}
		}
	}
		if(builder.length() > 1) {
			builder.deleteCharAt(builder.length() - 1);
		}
		builder.append("]");
		return builder.toString();
	}

	@Override
	public Iterator<TableEntry<K, V>> iterator() {
		return new IteratorImpl();
	}

	/**
	 * Razered koji modelira par kljuc-vrijednost u <code>SimpleHashtable</code> tablici
	 * @author Dominik Stipic
	 * @param <K> jedinstveni kljuc
	 * @param <V> vrijednost
	 */
	public static class TableEntry<K,V> {
		/**
		 * jedinstveni kljuc
		 */
		private K key;
		/**
		 * vrijednost 
		 */
		private V value;
		/**
		 * referenca na sljedeci <code>TableEntry</code>
		 */
		private TableEntry<K,V> next;
		
		/**
		 * Konstruktor koji stvara jedan par kljuc-vrijednost
		 * @param key kljuc 
		 * @param value vrijednost
		 * @param next referenca na sljedeci <code>TableEntry</code>
		 */
		public TableEntry(K key, V value, TableEntry<K,V> next) {
			this.key = Objects.requireNonNull(key, "Key can't be null");
			this.value = value;
			this.next = next;
		}

		/**
		 * Getter
		 * @return kljuc
		 */
		public K getKey() {
			return key;
		}

		/**
		 * Getter
		 * @return vrijednost
		 */
		public V getValue() {
			return value;
		}

		/**
		 * Setter
		 * @param next referenca na sljedeci par
		 */
		public void setNext(TableEntry<K,V> next) {
			this.next = next;
		}
		
		/**
		 * Setter
		 * @param value vrijednost
		 */
		public void setValue(V value) {
			this.value = value;
		}

		
		
		@Override
		public String toString() {
			return key + "=" + value;
		}

		@Override
		public int hashCode() {
			return key.hashCode();
		}
	}

	/**
	 * Razred koji zna iterirati po tablici.Vraca <code>TableEntry</code> i provjerava dali postoji sljedeci
	 * Prilikom iteririanja zabranjeno je mijenjanje tablice pomocu metoda koje nisu definirane unutar ovog razreda
	 * @author Dominik Stipic
	 *
	 */
	private class IteratorImpl implements Iterator<SimpleHashtable.TableEntry<K,V>>{
		/**
		 * broj slota koji se provjerava
		 */
		private int slotIndex = 0;
		/**
		 * sljedeci entry koji ce se vratiti
		 */
		private TableEntry<K, V> nextEntry;
		/**
		 * sadasnji entry
		 */
		private TableEntry<K, V> currentEntry;
		/**
		 * varijabla koja provjrava dali je doslo do modifikacija tablice dok traje iteracija
		 */
		private int modificationCountCompare;
		
		/**
		 * Konstruktor
		 */
		public IteratorImpl() {
			modificationCountCompare = modificationCount;
			if(size() == 0){
				currentEntry = null;
				nextEntry = null;
			} 
			else {
				while(table[slotIndex++] == null);
				nextEntry = table[--slotIndex];
				currentEntry = nextEntry;
			}
		}
		
		@Override
		public boolean hasNext() {
			if(modificationCountCompare != modificationCount) {
				throw new ConcurrentModificationException("A collection has been modified by non iterator methods");
			}
			return nextEntry != null;
		}

		@Override
		public TableEntry<K, V> next() {
			if(nextEntry == null) {
				throw new NoSuchElementException("No entries left in table");
			}
			if(modificationCountCompare != modificationCount) {
				throw new ConcurrentModificationException("A collection has been modified by non iterator methods");
			}
			currentEntry = nextEntry;
			nextEntry = findNext();
			return currentEntry;
		}
		
		@Override
		public void remove() {
			if(currentEntry == null) {
				throw new IllegalStateException("Cannot call remove method twice in row");
			}
			if(modificationCountCompare != modificationCount) {
				throw new ConcurrentModificationException("A collection has been modified by non iterator methods");
			}
			SimpleHashtable.this.remove(currentEntry.key);
			++modificationCountCompare;
			currentEntry = null;
		}

		/**
		 * Pronalazi sljedeci objekt od trenutnog <code>currentEntry</code>
		 * @return sljedeci <code>TableEntry</code> koji ce se vratiti
		 */
		private TableEntry<K, V> findNext() {
			if(slotIndex >= tableCapacity) {
				return null;
			}
			else if(currentEntry.next == null) {
				++slotIndex;
				while(slotIndex < tableCapacity && table[slotIndex] == null) {
					++slotIndex;
				}
				return slotIndex >= tableCapacity ? null : table[slotIndex];
			}
			else {
				return currentEntry.next;
			}
		}
		
		
		
	}
}
