package hr.fer.nenr.interfaces;

import java.util.List;

import Jama.Matrix;
import hr.fer.nenr.models.Parameters;


public interface ComputationalBlock {
	List<Double> forward(List<Double> inputs);
	List<Double> backward(List<Double> gradients);
	int inputFeatures();
	int outFeatures();
	void cleanGradients();
	Parameters getParams();
	void setParams(Parameters params);
	boolean hasParam();
	Matrix[] getGrads();
}
