package hr.fer.zemris.java.gui.layouts;

import java.util.Objects;

/**
 * Encapsulates information of button position on container layout
 * @author Dominik Stipić
 *
 */
public class RCPosition {
	/**
	 * row in {@link CalcLayout}
	 */
	private int row;
	/**
	 * colomn in {@link CalcLayout}
	 */
	private int colomn;
	/**
	 * max row
	 */
	public static final int MAX_ROW = 5;
	/**
	 * max colmon
	 */
	public static final int MAX_COLOMN = 7;
	
	
	/**
	 * Creates object which containes all information about comopnent position
	 * @param row index
	 * @param colomn index
	 * @throws CalcLayoutException - if provided with row or colomn which is out of bounds
	 */
	public RCPosition(int row, int colomn) {
		if(row < 0 || row > MAX_ROW || colomn < 0 && colomn > MAX_COLOMN) {
			throw new CalcLayoutException("illegal numeber of rows or colomns");
		}
		this.row = row;
		this.colomn = colomn;
	}


	/**
	 * gets the row
	 * @return row
	 */
	public int getRow() {
		return row;
	}


	/**
	 * gets the colomn
	 * @return colomn
	 */
	public int getColomn() {
		return colomn;
	}


	@Override
	public int hashCode() {
		return Objects.hash(row,colomn);
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		RCPosition other = (RCPosition) obj;
		if (colomn != other.colomn)
			return false;
		if (row != other.row)
			return false;
		return true;
	}
	
	
	
}
