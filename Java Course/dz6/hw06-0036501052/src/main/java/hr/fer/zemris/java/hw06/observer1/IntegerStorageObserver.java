package hr.fer.zemris.java.hw06.observer1;

/**
 * Interface which is implemented by classes which 
 * want to observe <code>IntegerStorage</code> changes 
 * @author Dominik Stipic
 *
 */
public interface IntegerStorageObserver {
	/**
	 * Gives the subscriber class information that observed class value
	 * have been changed.
	 * @param istorage New value which had been setted up in 
	 * observed class
	 */
	public void valueChanged(IntegerStorage istorage);
}