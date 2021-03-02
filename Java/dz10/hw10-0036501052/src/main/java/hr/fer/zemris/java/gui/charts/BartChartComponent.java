package hr.fer.zemris.java.gui.charts;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.geom.Line2D;
import java.util.List;
import static java.lang.Math.abs;

import javax.swing.JComponent;

/**
 * Component which can draw hisotgram form given data.
 * @author Dominik Stipić
 *
 */
public class BartChartComponent extends JComponent{
	private static final long serialVersionUID = 1L;
	/**
	 * gap beetween border and drawable space
	 */
	private static final int GAP = 36;
	/**
	 * space beetween string and axis
	 */
	private static final int NUM_TO_AXIS = 20;
	/**
	 * space beetween frame border and axis
	 */
	private static final int BORDER_TEXT = 7;
	/**
	 * chart info
	 */
	private BarChart chart;
	/**
	 * length of one x unit
	 */
	private int xLength;
	/**
	 * length of one y unit
	 */
	private int yLength;

	public BartChartComponent(BarChart chart) {
		this.chart = chart;
	}

	@Override
	protected void paintComponent(Graphics g) {
		Graphics2D g2 = (Graphics2D) g;
		drawTable(g2);
		drawAxis(g2);
		drawDescription(g2);
		drawData(g2);
	}
	
	/**
	 * draws the data in component as hisotgram
	 * @param g2 drawer
	 */
	private void drawData(Graphics2D g2) {
		int n = chart.getSize();
		g2.setColor(Color.ORANGE);
		List<XYValue> list = chart.getList();
		for(int i = 0; i < n; ++i ) {
			int height = abs((list.get(i).getY()/chart.getDeltaY()) * yLength);
			int width = xLength;
			int x = GAP + i * xLength ;
			int y = getHeight() - height - GAP;
			g2.fill3DRect(x, y, width, height,true);
		}
	}
	
	/**
	 * Draws table in conatiner space
	 * @param g2 drawer
	 */
	private void drawTable(Graphics2D g2){
		Dimension d = getSize();
		g2.setStroke(new BasicStroke(1));
		
		int deltaY = chart.getDeltaY();
		int maxY = chart.getMaxY();
		int n = maxY/deltaY;
		yLength = (d.height - GAP) / n;
		for(int i = 0; i <= n; ++i ) {
			int xStart = GAP;
			int xFinish = d.width;
			int y = (d.height-GAP) - (i)*yLength; 
			g2.draw(new Line2D.Double(xStart, y, xFinish, y));
		}
		
		n = chart.getSize();
		xLength = (d.width - GAP)/n;
		if(xLength % 2 == 1) ++xLength;
		for(int i = 0; i <= n; ++i ) {
			int yStart = 0;
			int yEnd = d.height - GAP;
			int x = GAP + i * xLength;
			g2.draw(new Line2D.Double(x, yStart, x, yEnd));
		}
	}
	
	/**
	 * draws x and y axis
	 * @param g2 drawer
	 */
	private void drawAxis(Graphics2D g2) {
		Dimension d = getSize();
		g2.setStroke(new BasicStroke(3.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL));
		
		int deltaY = chart.getDeltaY();
		int maxY = chart.getMaxY();
		int n = maxY/deltaY;
		for(int i = 0; i <= n; ++i ) {
			int x = GAP;
			int yStart = (d.height - GAP) - i*yLength;
			int yFinish = (d.height - GAP) - (i+1)*yLength;
			g2.draw(new Line2D.Double(x, yStart, x, yFinish));
			g2.draw(new Line2D.Double(x, yFinish, x - 5, yFinish));
			g2.drawString(String.valueOf(chart.getMinY() + i*deltaY), x - NUM_TO_AXIS, yStart);
			
		}
		
		n = chart.getSize();
		for(int i = 0; i < n; ++i ) {
			int xStr = (GAP + xLength/2) + xLength * i;
			int yStr = d.height - NUM_TO_AXIS;
			g2.drawString(String.valueOf(chart.getList().get(i).getX()), xStr, yStr);
		}
		g2.draw(new Line2D.Double(GAP, d.height - GAP, d.width, d.height - GAP));
		
	}
	
	/**
	 * draws string description neary axis
	 * @param g2 drawer
	 */
	private void drawDescription(Graphics2D g2) {
		String xStr = chart.getxDescription(); 
		String yStr = chart.getyDescription(); 
		int x = getSize().width/3 - xStr.length();
		int y = getSize().height - BORDER_TEXT;
		g2.drawString(chart.getxDescription(), x, y);
		
		AffineTransform at = new AffineTransform();
		at.rotate(-Math.PI / 2);
		Graphics2D g = (Graphics2D) g2.create();
		g.setTransform(at);
		
		x = getHeight()/2 - yStr.length();;
		y = BORDER_TEXT + 5;
		g.drawString(yStr, -x, y);
	}
}
