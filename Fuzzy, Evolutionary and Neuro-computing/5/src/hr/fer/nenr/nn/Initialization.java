package hr.fer.nenr.nn;

import java.util.function.Supplier;
import hr.fer.nenr.utils.Utils;

public class Initialization {
	
	public static Supplier<Double> uniform(int lower, int upper){
		Supplier<Double> supplier = () -> Utils.randomDouble(lower, upper);
		return supplier;
	}
}
