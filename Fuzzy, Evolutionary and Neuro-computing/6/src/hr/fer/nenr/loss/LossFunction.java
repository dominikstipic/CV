package hr.fer.nenr.loss;

import Jama.Matrix;

public interface LossFunction{
	double loss(Matrix out, Matrix target);
	Matrix backward();
}
