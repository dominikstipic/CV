package hr.fer.zemris.java.gui.layouts;

import static hr.fer.zemris.java.gui.calc.Operations.ADD;
import static hr.fer.zemris.java.gui.calc.Operations.DIV;
import static hr.fer.zemris.java.gui.calc.Operations.GET_COS;
import static hr.fer.zemris.java.gui.calc.Operations.GET_CTG;
import static hr.fer.zemris.java.gui.calc.Operations.GET_LN;
import static hr.fer.zemris.java.gui.calc.Operations.GET_LOG;
import static hr.fer.zemris.java.gui.calc.Operations.GET_SIN;
import static hr.fer.zemris.java.gui.calc.Operations.GET_TAN;
import static hr.fer.zemris.java.gui.calc.Operations.MUL;
import static hr.fer.zemris.java.gui.calc.Operations.SUB;
import java.awt.Color;
import java.awt.Font;
import java.awt.event.ActionListener;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.Stack;
import java.util.function.DoubleBinaryOperator;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

import hr.fer.zemris.java.gui.calc.CalcModel;
import hr.fer.zemris.java.gui.calc.CalcValueListener;
import hr.fer.zemris.java.gui.calc.Operations;

/**
 * Designs Calculator gui and assigns every calculator component its functionality.
 * @author Dominik Stipić
 *
 */
public class Calculator extends JFrame {
	private static final long serialVersionUID = 1L;

	/**
	 * Panel taht containes all components
	 */
	private JPanel panel = new JPanel(new CalcLayout(2));
	/**
	 * functionality model
	 */
	private CalcModel model = new CalcModelImpl();
	/**
	 * result display
	 */
	private Display display = new Display();
	/**
	 * number buttons
	 */
	private JButton numberButton[] = new JButton[10];
	/**
	 * stack for saving and manipulating operands.
	 */ 
	private Stack<Double> stack = new Stack<>();
	/**
	 * unary functions which have their inverse
	 */
	private Set<CalcButton> inversible = new HashSet<>();

	/**
	 * Creates calculator frame
	 */
	public Calculator() {
		setSize(600, 600);
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		setTitle("Ultra powerfull calculator");
		getContentPane().add(panel);
		initGUI();
	}

	/**
	 * initializes gui
	 */
	private void initGUI() {
		panel.add(display, "1,1");
		model.addCalcValueListener(display);

		insertNumberButtons();
		
		CalcButton decimal = new CalcButton(".");
		decimal.addActionListener( l -> model.insertDecimalPoint());
		panel.add(decimal, "5,4");

		addUnaryOperations();
		addBinaryOperations();
		addCommandOperations();
		
		JCheckBox inv = new JCheckBox("inv");
		inv.addActionListener(l->{
			if(inv.isEnabled())Operations.changeMode();
			inversible.forEach(btn -> btn.setText(btn.getSupplier().get()));
		});
		panel.add(inv,"5,7");
	}

	/**
	 * assigns functionality to command buttons
	 */
	private void addCommandOperations() {
		CalcButton equal = new CalcButton("=");
		equal.addActionListener(l->{
				Double first;
				try {
					first = model.getActiveOperand();
				} catch (Exception e1) {
					display.errorMessage("error");
					return;
				}
				Double second = model.getValue();
				DoubleBinaryOperator oper = model.getPendingBinaryOperation();
				try {
					model.setValue(oper.applyAsDouble(first, second));
				} catch (Exception e) {
					display.errorMessage(e.getMessage());
					return;
				}
				model.clearActiveOperand();
				model.setPendingBinaryOperation(null);
		});
		CalcButton clr =  new CalcButton("clr");
		clr.addActionListener(l -> model.clear());
		CalcButton res = new CalcButton("res");
		res.addActionListener(l-> model.clearAll());
		CalcButton push = new CalcButton("push");
		push.addActionListener(l->stack.push(model.getValue()));
		CalcButton pop = new CalcButton("pop");
		pop.addActionListener(l->{
			if(stack.isEmpty()) {
				display.errorMessage("stack is empty");
				return;
			}
			model.setValue(stack.pop());
		});
		
		panel.add(equal,"1,6");
		panel.add(clr,"1,7"); 
		panel.add(res,"2,7");
		panel.add(push,"3,7");
		panel.add(pop,"4,7");
	}

	/**
	 * assigns functionality to unary operation buttons
	 */
	private void addUnaryOperations() {
		CalcButton sin = new CalcButton(GET_SIN.get(), model, display);
		sin.setSupplier(GET_SIN);
		sin.setUnaryOperation();
		
		CalcButton cos = new CalcButton(GET_COS.get(), model, display);
		cos.setSupplier(GET_COS);
		cos.setUnaryOperation();
		
		CalcButton tan = new CalcButton(GET_TAN.get(), model, display);
		tan.setSupplier(GET_TAN);
		tan.setUnaryOperation();
		
		CalcButton ctg = new CalcButton(GET_CTG.get(), model, display);
		ctg.setSupplier(GET_CTG);
		ctg.setUnaryOperation();
		
		CalcButton log = new CalcButton(GET_LOG.get(), model, display);
		log.setSupplier(GET_LOG);
		log.setUnaryOperation();
		
		CalcButton ln = new CalcButton(GET_LN.get(), model, display);
		ln.setSupplier(GET_LN);
		ln.setUnaryOperation();
		
		CalcButton swap = new CalcButton("+/-");
		swap.addActionListener(l -> model.swapSign());
		CalcButton inv = new CalcButton("1/x", model, display);
		inv.setSupplier(Operations.GET_INV);
		inv.setUnaryOperation();
		
		inversible.addAll(Arrays.asList(sin,cos,tan,ctg,log,ln));
		
		panel.add(sin,"2,2");
		panel.add(cos,"3,2");
		panel.add(tan,"4,2");
		panel.add(ctg,"5,2");
		panel.add(inv,"2,1");
		panel.add(log,"3,1");
		panel.add(ln,"4,1");
		panel.add(swap,"5,5");
	}

	/**
	 * assigns functionality to binary operation buttons
	 */
	private void addBinaryOperations() {
		CalcButton plus = new CalcButton("+", model, display);
		plus.setBinaryOperation(ADD);
		CalcButton minus = new CalcButton("-", model, display);
		minus.setBinaryOperation(SUB);
		CalcButton mul = new CalcButton("*", model, display);
		mul.setBinaryOperation(MUL);
		CalcButton div = new CalcButton("/", model, display);
		div.setBinaryOperation(DIV);
		CalcButton nth = new CalcButton(Operations.GET_NPOW.get(),model, display);
		if(Operations.GET_NPOW.get().equals("x^n")) {
			nth.setBinaryOperation(Operations.POW_N);
		}
		else {
			nth.setBinaryOperation(Operations.ROOT_N);
		}
		
		panel.add(plus, "5,6");
		panel.add(minus, "4,6");
		panel.add(mul,"3,6");
		panel.add(div,"2,6");
		panel.add(nth,"5,1");
	}

	/**
	 * insertes number buttons in container
	 */
	private void insertNumberButtons() {
		ActionListener action = a -> {
			JButton b = (JButton) a.getSource();
			int digit = Integer.valueOf(b.getText());
			model.insertDigit(digit);
		};
		numberButton[0] = new CalcButton("0");
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				int index = (3 * i + j + 1);
				numberButton[index] = new CalcButton(String.valueOf(index));
				numberButton[index].addActionListener(action);
				int r = (4 - i);
				int c = (3 + j);
				panel.add(numberButton[index], new RCPosition(r, c));
			}
		}
		panel.add(numberButton[0], new RCPosition(5, 3));
		numberButton[0].addActionListener(action);
	}

	/**
	 * Method which automatically starts at program begining
	 * @param args
	 */
	public static void main(String[] args) {
		SwingUtilities.invokeLater(() -> {
			new Calculator().setVisible(true);
		});
	}

	/**
	 * Models {@link Calculator} display
	 * @author Dominik Stipić
	 *
	 */
	public static class Display extends JLabel implements CalcValueListener {
		private static final long serialVersionUID = 1L;

		/**
		 * Creates display as jlabel
		 */
		public Display() {
			setText("0");
			setBorder(BorderFactory.createLineBorder(Color.black, 2));
			setHorizontalAlignment(JLabel.RIGHT);
			setFont(new Font("Serif", Font.BOLD, 40));
			setOpaque(true);
			setBackground(Color.YELLOW);
		}

		@Override
		public void valueChanged(CalcModel model) {
			setText(model.toString());
		}
		
		/**
		 * error message which is going to be written on display
		 * @param error
		 */
		public void errorMessage(String error) {
			setText(error);
		}
	}

}
