package hr.fer.nenr.blocks;

import java.util.List;
import java.util.function.Function;

import Jama.Matrix;
import hr.fer.nenr.utils.MatrixAdapter;
import hr.fer.nenr.utils.VectorUtils;

public class Sigmoid extends Functional{
	private Matrix memory;
	private Function<Double, Double> function = d -> 1/(1+Math.exp(-d));
	
	public Sigmoid(int in) {
		super(in, in);
	}

	@Override
	public Matrix forward(Matrix inputs) {
		// inputs : Nx1;
		List<Double> xs = MatrixAdapter.fromVec(inputs);
		xs = VectorUtils.vectorUniFunction(xs, function);
		Matrix out = MatrixAdapter.toVec(xs);
		memory = out.copy();
		return out;
	}

	@Override
	public Matrix backward(Matrix gradients) {
		Matrix ones = MatrixAdapter.replicate(1, 1, memory.getColumnDimension());
		Matrix a = ones.minus(memory);
		Matrix result = MatrixAdapter.timesElementwise(memory, a);
		return result;
	}


	
}
