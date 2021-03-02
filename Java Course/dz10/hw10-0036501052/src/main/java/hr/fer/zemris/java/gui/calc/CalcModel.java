package hr.fer.zemris.java.gui.calc;

import java.util.function.DoubleBinaryOperator;


/**
 * Models all calculator functionality and all associated calculator behavior
 * @author Dominik Stipić
 *
 */
public interface CalcModel {
	/**
	 * Adds observer in model's internal list
	 * @param l listener
	 */
	void addCalcValueListener(CalcValueListener l);
	/**
	 * Removes observer from model's internal list
	 * @param l listener
	 */
	void removeCalcValueListener(CalcValueListener l);
	/**
	 * returns string representation of current calculator state
	 * @return calculator state as string
	 */
	String toString();
	/**
	 * returns numerical representation of current calculator state
	 * @return calculator state as double
	 */
	double getValue();
	/**
	 * Sets new calculatoe state 
	 * @param value double value
	 */
	void setValue(double value);
	/**
	 * clears current state
	 */
	void clear();
	/**
	 * Deletes all states that was memorized during processing
	 */
	void clearAll();
	/**
	 * swaps the sign of current value
	 */
	void swapSign();
	/**
	 * inserts decimal
	 */
	void insertDecimalPoint();
	/**
	 * inserts decimal value at the end of current value 
	 * @param digit which is going to be added
	 */
	void insertDigit(int digit);
	/**
	 * Checks if the active operand is set
	 * @return appropriate boolean value
	 */
	boolean isActiveOperandSet();
	/**
	 * Gets the active operand 
	 * @return active operand as double value
	 * @throws IllegalStateException if the operand is not set 
	 */
	double getActiveOperand();
	/**
	 * sets the active operand
	 * @param activeOperand
	 */
	void setActiveOperand(double activeOperand);
	/**
	 * clears current active operand
	 */
	void clearActiveOperand();
	/**
	 * gets the current pending binary operation
	 * @return operation
	 */
	DoubleBinaryOperator getPendingBinaryOperation();
	/**
	 * Sets the new Double Binary Operation
	 * @param op new operation
	 */
	void setPendingBinaryOperation(DoubleBinaryOperator op);
}