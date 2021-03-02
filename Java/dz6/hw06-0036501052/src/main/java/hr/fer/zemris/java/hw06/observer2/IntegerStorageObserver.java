package hr.fer.zemris.java.hw06.observer2;


/**
 * Interface which is implemented by classes which 
 * want to observe <code>IntegerStorage</code> changes 
 * @author Dominik Stipic
 *
 */
public interface IntegerStorageObserver {
	/**
	 * Gives the subscribed class information that observed class value
	 * have been changed.
	 * @param changeInfo Informations passed to subscribers
	 */
	public void valueChanged(IntegerStorageChange changeInfo);
}