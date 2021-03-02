package hr.fer.zemris.java.gui.charts;

import java.util.LinkedList;
import java.util.List;

/**
 * Represents all information need for {@link BartChartComponent} drawing.
 * @author Dominik Stipic
 *
 */
public class BarChart {
	/**
	 * list of data
	 */
	private List<XYValue> list = new LinkedList<>();
	/**
	 * descripiton on x axis
	 */
	private String xDescription;
	/**
	 * description of y axis
	 */
	private String yDescription;
	/**
	 * minimal y
	 */
	private int minY;
	/**
	 * maximal y
	 */
	private int maxY;
	/**
	 * difference beetween two y values
	 */
	private int deltaY;
	
	/**
	 * Creates this BarChart model
	 * @param list of data
	 * @param xDescription x axis text
	 * @param yDescription y axis text
	 * @param minY minimal y
	 * @param maxY maximal y
	 * @param deltaY difference beetween two y values
	 */
	public BarChart(List<XYValue> list, String xDescription, String yDescription, int minY, int maxY, int deltaY) {
		this.list = list;
		this.xDescription = xDescription;
		this.yDescription = yDescription;
		this.minY = minY;
		this.maxY = maxY;
		this.deltaY = deltaY;
	}

	/**
	 * gets the chart data 
	 * @return chart data
	 */
	public List<XYValue> getList() {
		return list;
	}

	/**
	 * X axis descripton
	 * @return text from x axis
	 */
	public String getxDescription() {
		return xDescription;
	}

	/**
	* Y axis descripton
	 * @return text from y axis
	 */
	public String getyDescription() {
		return yDescription;
	}

	/**
	 * gets minimal y
	 * @return minimal y
	 */
	public int getMinY() {
		return minY;
	}

	/**
	 * gets maximal y
	 * @return maximal y
	 */
	public int getMaxY() {
		return maxY;
	}

	/**
	 * gets the differnce beetwen y values
	 * @return differnce beetwen y values
	 */
	public int getDeltaY() {
		return deltaY;
	}
	
	/**
	 * number of chart data
	 * @return size od chart data
	 */
	public int getSize() {
		return list.size();
	}
	
}
