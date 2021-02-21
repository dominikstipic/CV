package hr.fer.nenr.graphic;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;

import javax.swing.BorderFactory;
import javax.swing.JPanel;

import hr.fer.nenr.interfaces.ISubscriber;
import hr.fer.nenr.models.GestureModel;

public class DrawArea extends JPanel implements ISubscriber{
	private static final long serialVersionUID = 1L;
	protected GestureModel gesture = new GestureModel();
	protected int stroke = 5;
	protected Color currentColor = Color.BLUE;
	
	
	public DrawArea() {
		setBorder(BorderFactory.createLineBorder(Color.BLACK));
		setPreferredSize(new Dimension(500, 500));
	}
	
	private void clear(Graphics2D g2) {
		Color color = g2.getColor();
		g2.setColor(Color.WHITE);
		g2.fillRect(0, 0, getWidth(), getHeight());
		g2.setColor(color);
	}
	
	private void configureGrahics(Graphics2D g2) {
		g2.setColor(currentColor);
		g2.setStroke(new BasicStroke(stroke));
	}

	private void drawGesture(Graphics2D g2) {
		for(int i = 0; i < gesture.size()-1; ++i) {
			Point first = gesture.get(i).point();
			Point second = gesture.get(i+1).point();
			int x1 = (int) first.getX();
			int y1 = (int) first.getY();
			int x2 = (int) second.getX();
			int y2 = (int) second.getY();
			g2.drawLine(x1, y1, x2, y2);
		}
	}
	
	@Override
	public void paintComponent(Graphics g) {
		Graphics2D g2 = (Graphics2D) g;
		clear(g2);
		configureGrahics(g2);
		drawGesture(g2);
	}
 
	@Override
	public void update(Object context) {
		gesture = (GestureModel) context;
		repaint();
	}
	
	
	
}
