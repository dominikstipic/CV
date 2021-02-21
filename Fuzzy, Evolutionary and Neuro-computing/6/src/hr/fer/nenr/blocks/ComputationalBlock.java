package hr.fer.nenr.blocks;

import java.util.List;

import Jama.Matrix;
import hr.fer.nenr.models.Parameters;


public interface ComputationalBlock {
	Matrix forward(Matrix inputs);
	Matrix backward(Matrix gradients);
	int inputFeatures();
	int outFeatures();
	void cleanGradients();
	List<Parameters> getParams();
	void setParams(List<Parameters> params);
	boolean hasParam();
}
