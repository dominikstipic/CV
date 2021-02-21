package hr.fer.nenr.blocks;

import Jama.Matrix;
import hr.fer.nenr.utils.MatrixAdapter;

public class NormalizationLayer extends Functional{
	private double sum;
	private Matrix memory;
	
	public NormalizationLayer(int in) {
		super(in,in);
	}
	
	@Override
	public Matrix forward(Matrix inputs) {
		// inputs : 1,N
		memory = inputs.copy();
		Matrix e = MatrixAdapter.replicate(1.0, in, 1);
		sum = inputs.times(e).get(0, 0);
		Matrix result = inputs.times(1/sum);
		return result;	
	}

	@Override
	public Matrix backward(Matrix gradients) {
		// gradients : 1/N
		int N = gradients.getColumnDimension();
		Matrix e = Matrix.identity(N, N).times(1/sum);
		Matrix dw = MatrixAdapter.tile(memory, true).times(1/(sum*sum));
		dw = e.minus(dw);
		
		Matrix result = gradients.times(dw);
		return result;
	}
	
	
	
	
	
	
	

	
}
