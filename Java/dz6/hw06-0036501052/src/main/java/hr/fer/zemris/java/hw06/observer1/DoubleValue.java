package hr.fer.zemris.java.hw06.observer1;

/**
 * Doubles the <code>IntegerStorageObserver</code> value when it changes.
 * Doubling occurs certain amount of times which is specified through constructor.
 * When doubling finishes class is automaticaly unregistreted from <code>IntegerStorageObserver</code> class
 * @author Dominik Stipic
 *
 */
public class DoubleValue  implements IntegerStorageObserver{
	/**
	 * number of times until end
	 */
	private int number;
	
	/**
	 * Constructor with number that specifies how 
	 * many times this class will double updated value
	 * @param number
	 */
	public DoubleValue (int number) {
		if(number < 0) {
			throw new IllegalArgumentException("Desired repeating number must be greater than 0");
		}
		this.number = number;
	}

	@Override
	public void valueChanged(IntegerStorage istorage) {
		if(number == 0) {
			istorage.removeObserver(this);
			return;
		}
		System.out.println("Double value: " + istorage.getValue() * 2);
		--number;
	}
	
	
	
}
