package hr.fer.nenr.blocks;

import Jama.Matrix;
import hr.fer.nenr.utils.MatrixAdapter;

public class Summer extends Functional{
	private Matrix memory;
	
	public Summer(int in) {
		super(in,1);
	}
	
	@Override
	public Matrix forward(Matrix inputs) {
		int n = inputs.getColumnDimension();
		memory = MatrixAdapter.replicate(1, 1, n);
		Matrix e = MatrixAdapter.replicate(1, 1, n);
		Matrix sum = inputs.times(e.transpose());
		return sum;
	}

	@Override
	public Matrix backward(Matrix gradients) {
		int rows = gradients.getRowDimension();
		int cols = gradients.getColumnDimension();
		if(rows != 1 | cols != 1)
			throw new IllegalArgumentException("input gradients should have the same dimension as output of summer");
		double g = gradients.get(0, 0);
		gradients = memory.times(g);
		return gradients;
	}

	

	

}
