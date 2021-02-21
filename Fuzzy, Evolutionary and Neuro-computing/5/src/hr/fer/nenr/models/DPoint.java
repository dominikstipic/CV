package hr.fer.nenr.models;

import java.awt.Point;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class DPoint {
	public double x;
	public double y;
	public DPoint(double x, double y) {
		this.x = x;
		this.y = y;
	}
	
	public static DPoint fromPoint(Point p) {
		return new DPoint(p.x, p.y);
	}
	
	public Point point() {
		return new Point((int)x,(int)y);
	}

	@Override
	public String toString() {
		return Arrays.asList(x,y).toString();
	}

	@Override
	public int hashCode() {
		return Objects.hash(x,y);
	}

	@Override
	public boolean equals(Object obj) {
		if(!obj.getClass().equals(DPoint.class)) return false;
		DPoint p = (DPoint) obj;
		return p.x == x && p.y == y;
	}
	
	
	
}
