package hr.fer.zemris.java.hw06.observer2;

/**
 * Squares the newly updated value in <code>IntgerStorage</code>
 * @author Dominik Stipic
 *
 */
public class SquareValue implements IntegerStorageObserver{
	@Override
	public void valueChanged(IntegerStorageChange istorage) {
		int value = istorage.getNewInt();
		System.out.println("Provided new value: " + value + ", square is " + value * value);
	}
}
