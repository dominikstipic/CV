package hr.fer.nenr.blocks;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import Jama.Matrix;
import hr.fer.nenr.utils.MatrixAdapter;

public class ProductWindow extends Functional{
	private Matrix[] memory;
	private int windowSize;
	
	public ProductWindow(int in, int out) {
		super(in, out);
		if(in % out != 0) 
			throw new IllegalArgumentException("In and out arent div");
		windowSize = in / out;
	}
	
	@Override
	public Matrix forward(Matrix inputs) {
		// inputs : 1,N
		if(!MatrixAdapter.isVector(inputs, true)) throw new IllegalArgumentException("input is not a row vec");
		memory = MatrixAdapter.windowing(inputs, windowSize);
		List<Double> w = new ArrayList<>();
		for(Matrix m : memory) {
			double p = MatrixAdapter.product(m);
			w.add(p);
		}
		return MatrixAdapter.toVec(w);
	}

	private Matrix reverseVector(Matrix X) {
		List<Double> xs = MatrixAdapter.fromVec(X);
		Collections.reverse(xs);
		Matrix revRow = MatrixAdapter.toVec(xs);
		return revRow;
	}
	
	@Override
	public Matrix backward(Matrix gradients) {
		// gradients : 1/N
		if(!MatrixAdapter.isVector(gradients, true)) throw new IllegalArgumentException("gradient is not a row vec");
		
		Matrix G = MatrixAdapter.replicate(0.0, out, in);
		for(int i = 0; i < memory.length; ++i) {
			Matrix row = memory[i];
			Matrix revRow = reverseVector(row);
			G.setMatrix(i, i, i*windowSize, (i+1)*windowSize-1, revRow);
		}
		Matrix result = gradients.times(G);
		return result;
	}
}