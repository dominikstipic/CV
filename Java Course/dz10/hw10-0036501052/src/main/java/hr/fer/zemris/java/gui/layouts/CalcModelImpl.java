package hr.fer.zemris.java.gui.layouts;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.function.DoubleBinaryOperator;

import hr.fer.zemris.java.gui.calc.CalcModel;
import hr.fer.zemris.java.gui.calc.CalcValueListener;

/**
 * Implementation of abstract calculator model and it functions.This model is 
 * combined with {@link Calculator} and includes all calculator functionality.
 * @author DOMINIK Stipić
 *
 */
public class CalcModelImpl implements CalcModel{
	/**
	 * value on "display"
	 */
	private String currentValue = "";
	/**
	 * registred listeners 
	 */
	private Set<CalcValueListener> listeners = new HashSet<>();
	/**
	 * math operation
	 */
	private DoubleBinaryOperator operation;
	/**
	 * active operand 
	 */
	private Double activeOperand;
	
	
	
	@Override
	public String toString() {
		if(currentValue.matches("(0+)?")) {
			return "0";
		}
		return currentValue;
	}

	@Override
	public void addCalcValueListener(CalcValueListener l) {
		Objects.requireNonNull(l, "listener cannot be null");
		listeners.add(l);
		
	}

	@Override
	public void removeCalcValueListener(CalcValueListener l) {
		Objects.requireNonNull(l, "listener cannot be null");
		listeners.remove(l);
	}

	/**
	 * updates registred listeners
	 */
	private void updateListeners() {
		Set<CalcValueListener> listenersCopy = new HashSet<>(listeners);
		for(CalcValueListener l: listenersCopy) {
			l.valueChanged(this);
		}
		listeners = listenersCopy;
	}
	
	@Override
	public double getValue() {
		if(currentValue.isEmpty())return 0.0;
		try {
			double d = Double.parseDouble(currentValue);
			return d;
		} catch (NumberFormatException e) {
			throw new IllegalArgumentException("input cannot be parsed to double -> " + currentValue);
		}
	}

	@Override
	public void setValue(double value) {
		if(Double.isNaN(value) || Double.isInfinite(value)) {
			throw new IllegalArgumentException("Invalid values ");
		}
		currentValue = String.valueOf(value);
		updateListeners();
	}

	@Override
	public void clear() {
		currentValue = "";
	}

	@Override
	public void clearAll() {
		currentValue = "";
		activeOperand = null;
		operation = null;
		updateListeners();
	}

	@Override
	public void swapSign() {
		if(!currentValue.isEmpty()) {
			double d = getValue() * -1;
			currentValue = String.valueOf(d);
			if(currentValue.matches("(-)?.\\.0"))currentValue = currentValue.substring(0,currentValue.length()-2);
			updateListeners();
		}
	}

	@Override
	public void insertDecimalPoint() {
		if(!currentValue.contains(".")) {
			currentValue += ".";
			updateListeners();
		}
	}

	@Override
	public void insertDigit(int digit) {
		if(currentValue.equals("0")  && digit == 0)return;
		if(currentValue.startsWith(".")) {
			currentValue = 0 + ".";
		}
		if(currentValue.equals("0") && digit != 0) {
			currentValue = String.valueOf(digit);
		}
		else {
			currentValue += digit;
			if(((Double)getValue()).isInfinite()) {
				currentValue = currentValue.substring(0, currentValue.length()-1);
			}
		}
		updateListeners();
	}

	@Override
	public boolean isActiveOperandSet() {
		return activeOperand != null;
	}

	@Override
	public double getActiveOperand() {
		if(activeOperand == null) {
			throw new IllegalStateException("Active operand wasn't set");
		}
		return activeOperand;
		
	}

	@Override
	public void setActiveOperand(double activeOperand) {
		this.activeOperand = activeOperand;
	}

	@Override
	public void clearActiveOperand() {
		activeOperand = null;
	}

	@Override
	public DoubleBinaryOperator getPendingBinaryOperation() {
		return operation;
	}

	@Override
	public void setPendingBinaryOperation(DoubleBinaryOperator op) {
		operation = op;
	}

}
