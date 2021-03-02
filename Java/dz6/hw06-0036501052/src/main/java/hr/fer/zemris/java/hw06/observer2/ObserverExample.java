package hr.fer.zemris.java.hw06.observer2;


/**
 * Demo which uses obserever design pattern
 * @author Dominik Stipic
 *
 */
public class ObserverExample {
	
	/**
	 * Method which is automaticaly started when program runs.
	 * @param args arguments from command line interface
	 */
	public static void main(String[] args) {
		IntegerStorage istorage = new IntegerStorage(20);
		istorage.addObserver(new DoubleValue(1));
		istorage.addObserver(new DoubleValue(2));
		istorage.addObserver(new DoubleValue(2));
		istorage.addObserver(new ChangeCounter());
		IntegerStorageObserver observer = new SquareValue();
		istorage.addObserver(observer);
		
		istorage.setValue(5);
		istorage.setValue(2);
		istorage.setValue(25);
		
		istorage.removeObserver(observer);
		
		istorage.setValue(13);
		istorage.setValue(22);
		istorage.setValue(15);
	}
}