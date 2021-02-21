package hr.fer.nenr.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import org.apache.commons.lang3.ArrayUtils;

public class LinAlg {
	
	private static void checkDim(double[] xs, double[] ys) {
		if(xs.length != ys.length) {
			throw new IllegalArgumentException("Vector dimensions should be the same");
		}
	}
	
	public static double[] vectorBiFunction(double[] a, double[] b, BiFunction<Double, Double, Double> function) {
		checkDim(a, b);
		int n = a.length;
		double []result = new double[n];
		for(int i = 0; i < n; ++i) {
			result[i] = function.apply(a[i], b[i]);
		}
		return result;
	}
	
	public static double[] vectorUniFunction(double[] a, Function<Double, Double> function) {
		int n = a.length;
		double []result = new double[n];
		for(int i = 0; i < n; ++i) {
			result[i] = function.apply(a[i]);
		}
		return result;
	}
	
	public static double[] minus(double[] a, double[] b) {
		BiFunction<Double, Double, Double> function = (x,y) -> x-y;
		return vectorBiFunction(a, b, function);
	}
	
	public static double[] plus(double[] a, double[] b) {
		BiFunction<Double, Double, Double> function = (x,y) -> x+y;
		return vectorBiFunction(a, b, function);
	}
	
	public static double[] div(double[] a, double[] b) {
		BiFunction<Double, Double, Double> function = (x,y) -> x/y;
		return vectorBiFunction(a, b, function);
	}
	
	public static double[] times(double[] a, double[] b) {
		BiFunction<Double, Double, Double> function = (x,y) -> x*y;
		return vectorBiFunction(a, b, function);
	}
	
	public static double scalarMul(double[] a, double[] b) {
		BiFunction<Double, Double, Double> function = (x,y) -> x*y;
		double[] result = vectorBiFunction(a, b, function);
		return sum(result);
	}
	
	public static double[] abs(double[] a) {
		Function<Double, Double> abs = x -> Math.abs(x);
		return vectorUniFunction(a, abs);
	}
	
	public static double[] pow(double[] a, int k) {
		Function<Double, Double> abs = x -> Math.pow(x,k);
		return vectorUniFunction(a, abs);
	}
	
	public static double[] sigm(double[] a) {
		Function<Double, Double> abs = x -> 1/(1+Math.exp(-x));
		return vectorUniFunction(a, abs);
	}
	
	public static double[] rep(double x, int n) {
		double[] xs = new double[n];
		for(int i = 0; i < n; ++i)xs[i] = x;
		return xs;
	}
	
	public static int[] rep(int x, int n) {
		int[] xs = new int[n];
		for(int i = 0; i < n; ++i)xs[i] = x;
		return xs;
	}
	
	public static double[] arange(int n) {
		double[] xs = new double[n];
		for(int i = 0; i < n; ++i)xs[i] = i;
		return xs;
	}
	
	public static double sum(double[] xs) {
		double sum = 0;
		for(Double s : xs) {
			sum += s;
		}
		return sum;
	}
	
	public static int sum(int[] xs) {
		int sum = 0;
		for(int s : xs) {
			sum += s;
		}
		return sum;
	}
	
	public static double norm(double[] xs) {
		xs = pow(xs, 2);
		double norm = sum(xs);
		return Math.sqrt(norm);
	}
	
	public static double[] vectorize(double[][] M) {
		List<Double> list = new ArrayList<>();
		for(int i = 0; i < M.length; ++i) {
			for(int j = 0; j < M[0].length; ++j) {
				list.add(M[i][j]);
			}
		}
		Double[] arr = list.toArray(new Double[0]);
		double[] res = ArrayUtils.toPrimitive(arr);
		return res;
	}
	
	public static double[] vectorize(Double[][] M) {
		double[][] m = new double[M.length][M[0].length];
		for(int i = 0;  i < M.length; ++i) {
			m[i] = ArrayUtils.toPrimitive(M[i]);
		}
		return vectorize(m);
	}
	
	public static double[][] t(double[][] M) {
		double[][] transposed = new double[M[0].length][M.length];
		for(int i = 0; i < M.length; ++i) {
			for(int j = 0; j < M[0].length; ++j) {
				transposed[j][i] = M[i][j];
			}
		}
		return transposed;
	}
	
	public static double[][] matrixMultiply(double[][] a, double[][] b) {
		double[][] result = new double[a.length][b[0].length];
		double [][] tb = t(b);
		int N = a.length;
		for(int i = 0; i < N; ++i) {
			double[] ai = a[i];
			for(int j = 0; j < N; ++j) {
				double[] bi = tb[j];
				double s = scalarMul(ai, bi);
				result[i][j] = s;
			}
		}
		return result;
	}
	
	public static int argmax(double[] xs) {
		double max = -Double.MIN_VALUE;
		int argmax = -1;
		for(int i = 0; i < xs.length; ++i) {
			if(xs[i] > max) {
				max = xs[i];
				argmax = i;
			}
		}
		return argmax;
	}
	
	public static int argmax(int[] xs) {
		double max = -Double.MIN_VALUE;
		int argmax = -1;
		for(int i = 0; i < xs.length; ++i) {
			if(xs[i] > max) {
				max = xs[i];
				argmax = i;
			}
		}
		return argmax;
	}
	
}
