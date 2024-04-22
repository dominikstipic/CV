package hr.fer.nenr.main;

import java.util.Arrays;
import java.util.List;

import Jama.Matrix;
import hr.fer.nenr.utils.MatrixAdapter;

public class Main1{
	
	
	public static void main(String[] args) {
		Matrix M = MatrixAdapter.symetricUniform(2, 3, 3);
		M.print(3, 3);
	}

	
	
}
