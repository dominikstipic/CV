package hr.fer.nenr.main;

import Jama.Matrix;
import hr.fer.nenr.utils.MatrixAdapter;

public class Main2 {

	private static Matrix expandInput(Matrix inputs, int rules) {
		Matrix result = inputs.copy();
		for(int i = 1; i < rules; ++i) {
			result = MatrixAdapter.stack(result, inputs, true);
		}
		return result;
	}
	
	public static void main(String[] args) {
		System.out.println(1E-4);
	}
}
