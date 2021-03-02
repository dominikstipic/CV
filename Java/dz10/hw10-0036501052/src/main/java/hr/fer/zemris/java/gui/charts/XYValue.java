package hr.fer.zemris.java.gui.charts;

/**
 * Encapsulates x and y values into one class
 * @author DOminik Stipić
 *
 */
public class XYValue {
	/**
	 * x value
	 */
	private int x;
	/**
	 * y value
	 */
	private int y;
	
	/**
	 * Saves X and Y value
	 * @param x value
	 * @param y value
	 */
	public XYValue(int x, int y) {
		this.x = x;
		this.y = y;
	}

	/**
	 * getter for x value
	 * @return x value
	 */
	public int getX() {
		return x;
	}

	/**
	 * getter for y value
	 * @return y value
	 */
	public int getY() {
		return y;
	}
	
}
