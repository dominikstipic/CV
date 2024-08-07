package hr.fer.zemris.java.hw06.observer1;

/**
 * Observer who counts the number of <code>IntegerStorageObserver</code>
 * changes.
 * @author Dominik Stipic
 *
 */
public class ChangeCounter implements IntegerStorageObserver{
	/**
	 * counts the numeber of changes
	 */
	int count = 0;
	
	@Override
	public void valueChanged(IntegerStorage istorage) {
		System.out.println("Number of value changes since tracking: " + (++count));
	}
}
