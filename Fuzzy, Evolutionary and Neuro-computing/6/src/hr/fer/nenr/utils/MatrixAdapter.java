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
	
	public static Matrix[] getRows(Matrix X) {
		int row = X.getRowDimension();
		int cols = X.getColumnDimension();
		Matrix[] rows = new Matrix[row];
		for(int i = 0; i < row; ++i) {
			rows[i] = X.getMatrix(i, i, 0, cols-1);
		}
		return rows;
	}
	
	public static void print(Matrix... ms) {
		for(Matrix m : ms)
			System.out.println(matrixString(m));
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
	
	public static Matrix toVec(Number... xs) {
		double[] arr = new double[xs.length];
		for(int i = 0; i < xs.length; ++i) {
			Number d = xs[i];
			arr[i] = d.doubleValue();
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
	
	public static Matrix timesElementwise(Matrix a, Matrix b) {
		if(a.getColumnDimension() != b.getColumnDimension()) throw new IllegalArgumentException("Dimensions doesn't match");
		if(a.getRowDimension() != 1 && b.getRowDimension() != 1) throw new IllegalArgumentException("function only operates with row vectors");
		int N = a.getColumnDimension();
		Matrix r = new Matrix(1, N);
		for(int i = 0; i < N; ++i) {
			double x = a.get(0, i);
			double y = b.get(0, i);
			r.set(0, i, x*y);
		}
		return r;
	}
	
	public static Matrix tile(Matrix a, boolean axis) {
		if(a.getRowDimension() != 1) throw new IllegalArgumentException("function only operates with row vectors");
		int N = a.getColumnDimension();
		Matrix e = replicate(1.0, 1, N);
		Matrix result = a.transpose().times(e);
		if(!axis) result = result.transpose();
		return result;
		
	}
	
	public static Matrix permutationMatrix(int x, int y, int N) {
		Matrix E = replicate(0.0, N, N);
		E.set(y, x, 1.0);
		E.set(x, y, 1.0);
		return E;
	}
	
	public static Matrix permute(Matrix X, int x, int y) {
		int N = X.getRowDimension();
		Matrix P = permutationMatrix(x, y, N);
		Matrix result = X.times(P);
		return result;
	}
	
	public static Matrix diag(Matrix vector) {
		int N = vector.getColumnDimension();
		Matrix E = Matrix.identity(N, N);
		for(int i = 0; i < N; ++i) {
			double val = vector.get(0, i);
			E.set(i, i, val);
		}
		return E;
	}
	
	public static Matrix stack(Matrix a, Matrix b, boolean axis) {
		int rowA = a.getRowDimension(); 
		int rowB = b.getRowDimension();
		int colA = a.getColumnDimension();
		int colB = b.getColumnDimension();
		Matrix result;
		if(axis & rowA != rowB) throw new IllegalArgumentException("rows doesn't match");
		if(!axis & colA != colB) throw new IllegalArgumentException("cols doesn't match");
		
		if(axis) {
			result = new Matrix(rowA, colA + colB);
			result.setMatrix(0, rowA-1, 0, colA-1, a);
			result.setMatrix(0,rowA-1, colA, colA + colB-1, b);
		}
		else {
			result = new Matrix(rowA + rowB, colA);
			result.setMatrix(0, rowA-1, 0, colA-1, a);
			result.setMatrix(rowA, rowA + rowB - 1, 0, colA-1, b);
		}
		return result;
	}
	
	public static boolean isVector(Matrix m, boolean rowVector) {
		int rows = m.getRowDimension();
		int cols = m.getColumnDimension();
		if(rowVector) {
			return rows == 1 && cols >= 1;
		}
		else{
			return rows >= 1 && cols == 1;
		}
	}
	
	public static boolean checkDim(Matrix a, Matrix b) {
		int rowA = a.getRowDimension(); 
		int rowB = b.getRowDimension();
		int colA = a.getColumnDimension();
		int colB = b.getColumnDimension();
		return rowA == rowB && colA == colB;
	}

	public static Matrix[] split(Matrix a, int m) {
		Matrix[] ms = new Matrix[m];
		int N = a.getColumnDimension();
		int d = N/m;
		for(int i = 0; i < m; ++i) {
			Matrix K = a.getMatrix(0, 0, i*d, (i+1)*d-1);
			ms[i] = K;
		}
		return ms;
	}
	
	public static Matrix[] windowing(Matrix M, int winSize) {
		int N = M.getColumnDimension();
		int groupNum = N / winSize;
		Matrix[] ms = new Matrix[groupNum];
		for(int i = 0; i < groupNum; ++i) {
			Matrix g = M.getMatrix(0, 0, i*winSize, (i+1)*winSize - 1);
			ms[i] = g;
		}
		return ms;
	}
	
	public static Matrix tile(Matrix X, int n) {
		Matrix row = X.copy();
		for(int i = 0; i < n-1; ++i) {
			X = MatrixAdapter.stack(X, row, false);
		}
		return X;
	}
	
	public static double product(Matrix M ) {
		int N = M.getColumnDimension();
		double product = 1;
		for(int i = 0; i < N; ++i) {
			product *= M.get(0, i);
		}
		return product;
	}
	
}
