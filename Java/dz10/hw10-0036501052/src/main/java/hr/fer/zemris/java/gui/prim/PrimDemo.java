package hr.fer.zemris.java.gui.prim;

import java.awt.BorderLayout;
import java.awt.GridLayout;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.SwingUtilities;

/**
 * Demonstrates usage of List model by creating appropriate gui.
 * List model generates prime numbers
 * @author DOMINIK Stipic
 *
 */
public class PrimDemo extends JFrame{
	private static final long serialVersionUID = 1L;
	/**
	 * btn for generatin new primes
	 */
	JButton btn = new JButton("sljedeći");

	/**
	 * Constructs gui
	 */
	public PrimDemo() {
		setTitle("PrimeDemo");
		setSize(600, 600);
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		initGui();
	}
	
	/**
	 * Gui builder
	 */
	private void initGui() {
		JPanel p = new JPanel();
		p.setLayout(new GridLayout(1, 2));
		getContentPane().add(p,BorderLayout.CENTER);
		
		PrimListModel model = new PrimListModel();
		JList<Integer> list1 = new JList<>(model);
		JList<Integer> list2 = new JList<>(model);
		
		p.add(new JScrollPane(list1));
		p.add(new JScrollPane(list2));
		
		btn.addActionListener(l->{
			model.next();
		});
		
		getContentPane().add(btn,BorderLayout.SOUTH);
	}
	
	/**
	 * Automatically starts
	 * @param args
	 */
	public static void main(String[] args) {
		SwingUtilities.invokeLater(()->{
			new PrimDemo().setVisible(true);
		});
	}
}
