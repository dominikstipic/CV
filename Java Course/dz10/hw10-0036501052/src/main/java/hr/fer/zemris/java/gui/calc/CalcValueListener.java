package hr.fer.zemris.java.gui.calc;

/**
 * Models objects which are intrested in calculator and wants to be notified about it's change
 * @author Dominik Stipić
 *
 */
public interface CalcValueListener {
	/**
	 * Notifies all calculator observers
	 * @param model of calculator
	 */
	void valueChanged(CalcModel model);
}