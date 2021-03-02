package hr.fer.zemris.java.hw06.observer1;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;

/**
 * Stores Integer value and can be "subscribed" to.
 * Subscibed class will be updated when this class changes. 
 * @author Dominik Stipic
 *
 */
public class IntegerStorage {
	/**
	 * storage value
	 */
	private int value;
	/**
	 * list of "subscribers"
	 */
	private List<IntegerStorageObserver> observers; 

	/**
	 * Constructor with initialValue which is going to be 
	 * stored
	 * @param initialValue value which is going to be stored
	 */
	public IntegerStorage(int initialValue) {
		this.value = initialValue;
		observers = new ArrayList<>();
	}

	/**
	 * Adds observer in the internal subscriber list 
	 * @param observer Subscriber to this class
	 */
	public void addObserver(IntegerStorageObserver observer) {
		Objects.requireNonNull(observer);
		observers.add(observer);
	}

	/**
	 * Removes observer from subscribers list
	 * @param observer which is going to be removed
	 * 
	 */
	public void removeObserver(IntegerStorageObserver observer) {
		Objects.requireNonNull(observer);
		observers.remove(observer);
	}

	/**
	 * Clears a list of subscribers
	 */
	public void clearObservers() {
		observers.clear();
	}

	/**
	 * Gets the storage value
	 * @return storage value
	 */
	public int getValue() {
		return value;
	}

	/**
	 * Sets the storage value 
	 * @param value which is going to be stored
	 */
	public void setValue(int value) {
		// Only if new value is different than the current value:
		if (this.value != value) {
			// Update current value
			this.value = value;
			// Notify all registered observers
			if (observers != null) {
				List<IntegerStorageObserver> observersCopy = new ArrayList<>(observers); // prevents concurrency modification
				for (IntegerStorageObserver observer : observersCopy) {
					observer.valueChanged(this);
				}
			}
		}
	}
}