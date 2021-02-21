package hr.fer.nenr.utils;

import java.util.ArrayList;
import java.util.List;
import Jama.Matrix;

public class MatrixAdapter {

	public static String matrixString(Matrix m) {
	    StringBuffer sb = new StringBuffer();
	    for (int r = 0; r < m.getRowDimension(); ++ r) {
	        for (int c = 0; c < m.getColumnDimension(); ++c)
	            sb.append(m.get(r, c)).append(" ");
	        sb.append("\n");
	    }
	    return sb.toString();
	}
	
	public static Matrix toVec(List<Double> xs) {
		double[] arr = new double[xs.size()];
		for(int i = 0; i < xs.size(); ++i) {
			double d = xs.get(i);
			arr[i] = d;
		}
		Matrix m = new Matrix(arr, 1);
		return m;
	}
	
	public static Matrix toMatrix(List<List<Double>> xs) {
		int cols = xs.get(0).size();
		int rows = xs.size();
		double[][] arr = new double[rows][cols];
		for(int i = 0; i < cols; ++i) {
			for(int j = 0; j < cols; ++j) {
				arr[i][j] = xs.get(i).get(j);
			}
		}
		Matrix m = new Matrix(arr);
		return m;
	}
	
	
	public static List<Double> fromVec(Matrix m) {
		double[][] d = m.getArray();
		List<Double> list = new ArrayList<>();
		int size = d[0].length;
		for(int i = 0; i < size; ++i) {
			list.add(d[0][i]);
		}
		return list;
	}
	
	public static List<List<Double>> fromMatrix(Matrix m) {
		int cols = m.getColumnDimension();
		int rows = m.getRowDimension();
		double[][] arr = m.getArray();
		List<List<Double>> list = new ArrayList<>();
		for(int i = 0; i < rows; ++i) {
			List<Double> xs = new ArrayList<>();
			for(int j = 0; j < cols; ++j) {
				double d = arr[i][j];
				xs.add(d);
			}
			list.add(xs);
		}
		return list;
	}
	
	public static Matrix replicate(double x, int rows, int cols) {
		double[][] arr = new double[rows][cols];
		for(int i = 0; i < rows; ++i) {
			for(int j = 0; j < cols; ++j) {
				arr[i][j] = x;
			}
		}
		Matrix M = new Matrix(arr);
		return M;
	}
	
	
	public static Matrix symetricUniform(double bound, int rows, int cols) {
		Matrix M = Matrix.random(rows, cols);
		M = M.times(bound*2);
		Matrix I = MatrixAdapter.replicate(bound, rows, cols);
		M = M.minus(I);
		return M;
	}
}
