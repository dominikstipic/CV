package hr.fer.zemris.java.hw06.observer1;

/**
 * Squares the newly setted value in <code>IntgerStorage</code>
 * @author Dominik Stipic
 *
 */
public class SquareValue implements IntegerStorageObserver{
	@Override
	public void valueChanged(IntegerStorage istorage) {
		int value = istorage .getValue();
		System.out.println("Provided new value: " + value + ", square is " + value * value);
	}
}
