package hr.fer.zemris.java.gui.layouts;


import java.awt.Dimension;

import javax.swing.JLabel;
import javax.swing.JPanel;

import org.junit.Assert;
import org.junit.Test;

public class CalcLayoutTest {

	@Test
	public void forPrefferedSize0() {
		JPanel p = new JPanel(new CalcLayout(2));
		JLabel l1 = new JLabel(""); l1.setPreferredSize(new Dimension(10,30));
		JLabel l2 = new JLabel(""); l2.setPreferredSize(new Dimension(20,15));
		p.add(l1, "2,2");
		p.add(l2, "3,3");
		Dimension dim = p.getPreferredSize();
		Assert.assertEquals(152, (int)dim.getWidth());
		Assert.assertEquals(158, (int)dim.getHeight());
		
	}
	
	@Test
	public void forPrefferedSize1() {
		JPanel p = new JPanel(new CalcLayout(2));
		JLabel l1 = new JLabel(""); l1.setPreferredSize(new Dimension(108,15));
		JLabel l2 = new JLabel(""); l2.setPreferredSize(new Dimension(16,30));
		p.add(l1, "1,1");
		p.add(l2, "3,3");
		Dimension dim = p.getPreferredSize();
		Assert.assertEquals(158, (int)dim.getHeight());
		Assert.assertEquals(152, (int)dim.getWidth());
		
	}
	
}
