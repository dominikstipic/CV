package nenr.main;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import nenr.dataset.IDataset;
import nenr.dataset.Measure;

public class Utils {

	public static double[] randDouble(int n, double lower, double upper) {
		double delta = upper - lower;
		double ds[] = new double[n];
		for(int i = 0; i < n; ++i) {
			long time = System.nanoTime();
			ds[i] = new Random(time).nextDouble()*delta + lower;
		}
		return ds;
	}
	
	public static boolean throwCoin(double prob) {
		long time = System.nanoTime();
		double value = new Random(time).nextDouble();
		return value < prob;
	}
	
	public static List<Double> arrange(int n){
		List<Double> xs = new ArrayList<>();
		for(int i = 0; i < n; ++i) {
			xs.add((double)i);
		}
		return xs;
	}
	
	public static void printDataset(IDataset dataset) {
		for(Measure m : dataset) {
			System.out.println(m);
		}
	}
	
	public static double[] toPrimitive(List<Double> list) {
		int n = list.size();
		double[] doubles = new double[n];
		for(int i = 0; i < list.size(); ++i) {
			doubles[i] = list.get(i);
		}
		return doubles;
	}
}
