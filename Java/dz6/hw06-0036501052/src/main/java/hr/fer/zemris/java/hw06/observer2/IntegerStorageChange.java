package hr.fer.zemris.java.hw06.observer2;

import java.util.Objects;

/**
 * Encapsulates information when change in <code>IntegerStorage</code>
 * occurs. 
 * @author Dominik Stipic
 *
 */
public class IntegerStorageChange {
	/**
	 * class which changed
	 */
	private IntegerStorage cause;
	/**
	 * old stored value
	 */
	private int oldInt;
	/**
	 * new stored value
	 */
	private int newInt;
	
	/**
	 * Creates class which encapsultes information about change
	 * @param cause Of change
	 * @param oldInt old storage 
	 * @param newInt new storage
	 */
	public IntegerStorageChange(IntegerStorage cause, int oldInt, int newInt) {
		this.cause = Objects.requireNonNull(cause, "IntegerStorage can not be null");
		this.oldInt = oldInt;
		this.newInt = newInt;
	}
	/**
	 * Gets the cause of change
	 * @return cause of change
	 */
	public IntegerStorage getCause() {
		return cause;
	}
	/**
	 * Gets the old integer value
	 * @return old storred Integer
	 */
	public int getOldInt() {
		return oldInt;
	}
	/**
	 * Gets the new integer value
	 * @return newly Storred Integer
	 */
	public int getNewInt() {
		return newInt;
	}
}
