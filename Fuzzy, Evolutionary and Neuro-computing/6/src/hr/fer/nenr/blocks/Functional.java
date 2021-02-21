package hr.fer.nenr.blocks;

import java.util.List;

import Jama.Matrix;
import hr.fer.nenr.models.Parameters;

public abstract class Functional implements ComputationalBlock{
	protected int in;
	protected int out;
	
	public Functional(int in, int out) {
		this.in = in;
		this.out = out;
	}

	@Override
	public boolean hasParam() {
		return false;
	}
	
	@Override
	public void cleanGradients() {
	}

	@Override
	public List<Parameters> getParams() {
		throw new UnsupportedOperationException("Component doesn't have parameters");
	}

	@Override
	public void setParams(List<Parameters> params) {
		throw new UnsupportedOperationException("Component doesn't have parameters");
	}

	@Override
	public abstract Matrix forward(Matrix inputs);

	@Override
	public abstract Matrix backward(Matrix gradients);

	@Override
	public int inputFeatures() {
		return in;
	}

	@Override
	public int outFeatures() {
		return out;
	}

	
	
	
}
