package hr.fer.nenr.loss;

import Jama.Matrix;

public class MSE implements LossFunction{
	private Matrix e;

	@Override
	public double loss(Matrix out, Matrix target) {
		this.e = target.minus(out);
		Matrix eSquared = e.times(e.transpose());
		Matrix result = eSquared.times(0.5);
		return result.get(0, 0);
	}

	@Override
	public Matrix backward() {
		Matrix result = e.times(-1);
		return result;
	}

	
	
	
}
