package hr.fer.nenr.utils;

import java.util.List;
import hr.fer.nenr.models.DPoint;
import static java.lang.Math.*;

public class PointUtils {

	public static DPoint mean(List<DPoint> points) {
		if(points.size() == 0) throw new IllegalArgumentException("size is 0");
		int N = points.size();
		int sumX = 0;
		int sumY = 0;
		
		for(DPoint p : points) {
			sumX += p.x;
			sumY += p.y;
		}
		
		double meanX = sumX/N;
		double meanY = sumY/N;
		return new DPoint(meanX, meanY);
	}
	
	public static double euclidanDistance(DPoint a, DPoint b) {
		double r = pow(a.x - b.x, 2) + pow(a.y - b.y, 2);
		r = sqrt(r);
		return r;
	}
	
	
}
