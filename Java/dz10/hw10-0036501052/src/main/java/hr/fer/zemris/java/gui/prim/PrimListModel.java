package hr.fer.zemris.java.gui.prim;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import javax.swing.ListModel;
import javax.swing.event.ListDataEvent;
import javax.swing.event.ListDataListener;

/**
 * Models ListModel which is used in JList.
 * This model generates prime numbers 
 * @author Dominik Stipić
 *
 */
public class PrimListModel implements ListModel<Integer>{
	/**
	 * Set of registated listeners
	 */
	private Set<ListDataListener> listeners = new HashSet<>();
	/**
	 * prime collection
	 */
	private List<Integer> primeNumbers = new LinkedList<>();
	{
		primeNumbers.add(1);
	}
	/**
	 * last prime
	 */
	private int prime = 1;
	
	/**
	 * generates new prime number
	 * @return prime number
	 */
	public int next() {
		setPrime();
		primeNumbers.add(prime);
		updateListeners();
		return prime;
	}
	
	/**
	 * sets the prime as atrbiute
	 */
	private void setPrime() {
		if(prime == 1) {
			++prime;
			return;
		}
		while(true) {
			++prime;
			boolean flag = true;
			for(int i = 2; i < prime; ++i) {
				if(prime % i == 0) {
					flag = false;
					break;
				}
			}
			if(flag == true) {
				break;
			}
		} 
	}
	
	@Override
	public void addListDataListener(ListDataListener l) {
		listeners.add(l);
	}

	@Override
	public Integer getElementAt(int index) {
		if(index > primeNumbers.size() || index < 0 ) {
			throw new IllegalArgumentException("index out of bounds");
		}
		return primeNumbers.get(index);
	}

	@Override
	public int getSize() {
		return primeNumbers.size();
	}

	@Override
	public void removeListDataListener(ListDataListener l) {
		listeners.remove(l);
	}

	/**
	 * updates listeners about change
	 */
	private void updateListeners () {
		ListDataEvent event = new ListDataEvent(this, ListDataEvent.INTERVAL_ADDED, primeNumbers.size()-1, primeNumbers.size()-1);
		Set <ListDataListener> setCopy = new HashSet<>(listeners);
		setCopy.forEach(l -> l.contentsChanged(event));
		listeners = setCopy;
	}
}
