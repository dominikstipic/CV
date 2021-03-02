package hr.fer.zemris.java.gui.layouts;


import java.awt.Font;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;

import javax.swing.JButton;

import hr.fer.zemris.java.gui.calc.CalcModel;
import hr.fer.zemris.java.gui.calc.Operations;
import hr.fer.zemris.java.gui.layouts.Calculator.Display;

/**
 * Represents button used as calculator button.
 * Contains text name and provides methods which updates
 * calculator state.
 * @author Dominik Stipić
 *
 */
public class CalcButton extends JButton {
	private static final long serialVersionUID = 1L;
	/**
	 * calculator 
	 */
	private CalcModel model;
	/**
	 * display of calculator
	 */
	private Display display;
	/**
	 * supplier for unary operations
	 */
	private Supplier<String> supplier;

	/**
	 * Creates button and saves provided arguments as internal atributtes
	 * @param text text of button
	 * @param model of calculator
	 * @param display of calculator
	 */
	public CalcButton(String text, CalcModel model,Display display) {
		this(text);
		this.model = model;
		this.display = display;
	}
	
	/**
	 * Constructs button and gives it a name associated with given string
	 * @param text name of calc
	 */
	public CalcButton (String text) {
		setText(text);
		setFont((new Font("Serif", Font.BOLD, 20)));
	}
	
	/**
	 * sets the binary operation action on this button 
	 * @param oper operation type
	 */
	public void setBinaryOperation(DoubleBinaryOperator oper) {
		addActionListener(l->{
			if(model.getPendingBinaryOperation() == null) {
				model.setActiveOperand(model.getValue());
			}
			else {
				Double res;
				try {
					res = model.getPendingBinaryOperation().applyAsDouble(model.getActiveOperand(), model.getValue());
					model.setValue(res);
					model.setActiveOperand(res);
				} catch (Exception e) {
					display.errorMessage("error");
				}
			}
			model.setPendingBinaryOperation(oper);
			model.clear();
		});
	}
	
	/**
	 * sets unary operation action on this button
	 */
	public void setUnaryOperation() {
		addActionListener(l->{
			Double res;
			try {
				DoubleUnaryOperator oper = Operations.getUnaryOperation(supplier.get());
				res = oper.applyAsDouble(model.getValue());
				model.setValue(res);
			} catch (Exception e) {
				display.errorMessage("error");
			}
		});
	}
	
	/**
	 * supplier getter
	 * @return supplier for unary operations
	 */
	public Supplier<String> getSupplier() {
		return supplier;
	}

	/**
	 * sets unary operation 
	 * @param supplier supplier for unary operations
	 */
	public void setSupplier(Supplier<String> supplier) {
		this.supplier = supplier;
	}
}