package hr.fer.nenr.utils;

import java.util.Random;

public class Utils {

	public static double randomDouble(double lower, double upper) {
		Random r = new Random(System.nanoTime());
		Double d = r.nextDouble()*upper-lower;
		return d;
	}
	
}
