package hr.fer.nenr.dataset;

import java.util.function.BiFunction;
import static java.lang.Math.*;

public class Functions {
	public static final BiFunction<Double, Double, Double> FUNCTION1 = (x,y) ->  (pow(x-1,2) + pow(y+2,2) - 5*x*y + 3) * pow(cos(x/5),2);
}
