package hr.fer.nenr.blocks;

import java.util.List;

import Jama.Matrix;
import hr.fer.nenr.interfaces.ComputationalBlock;
import hr.fer.nenr.models.Parameters;
import hr.fer.nenr.utils.MatrixAdapter;

public class Linear implements ComputationalBlock{
	private Matrix memory;
	private Matrix W;
	private Matrix biases;
	private Matrix weightGrads; //out, in
	private Matrix biasGrads;   // 1, out
	private int in; 
	private int out;
	
	public Linear(int inFeatures, int outFeatures) {
		in = inFeatures; 
		out = outFeatures;
		cleanGradients();
		initParams();
	}
	
	public void initParams() {
		W = MatrixAdapter.symetricUniform(1, out, in);
		biases = MatrixAdapter.symetricUniform(1, 1, out);
	}
	
	@Override
	public List<Double> forward(List<Double> inputs) {
		Matrix X = MatrixAdapter.toVec(inputs); // 1, N
		memory = X;  // 1, M
		Matrix S = W.times(X.transpose());
		S = S.transpose();
		S = S.plus(biases);
		List<Double> results = MatrixAdapter.fromVec(S);
		return results;
	}

	@Override
	public List<Double> backward(List<Double> gradients) {
		Matrix E = MatrixAdapter.toVec(gradients).transpose();  //N,1
		Matrix dw = E.times(memory); 
		accumulateGrads(dw, E.transpose());
		Matrix dx = W.transpose().times(E);
		dx = dx.transpose();
		List<Double> inputGrads = MatrixAdapter.fromVec(dx);
		return inputGrads;
	}

	@Override
	public void cleanGradients() {
		weightGrads = MatrixAdapter.replicate(0, out, in);
		biasGrads = MatrixAdapter.replicate(0, 1, out);
	}

	@Override
	public Parameters getParams() {
		Parameters params = new Parameters(W, biases);
		return params;
	}
	
	private boolean dimensionalityCheck(Matrix a, Matrix b) {
		boolean rows = a.getRowDimension() == b.getRowDimension();
		boolean cols = a.getColumnDimension() == b.getColumnDimension();
		return rows && cols;
	}

	@Override
	public void setParams(Parameters params) {
		if(!(dimensionalityCheck(params.W, W) && dimensionalityCheck(params.b, biases)))
			throw new IllegalArgumentException("Dimensions doesnt correspond");
		W = params.W;
		biases = params.b;
	}

	@Override
	public boolean hasParam() {
		return true;
	}

	@Override
	public Matrix[] getGrads() {
		Matrix[] arr = {weightGrads, biasGrads};
		return arr;
	}

	@Override
	public int inputFeatures() {
		return in;
	}

	@Override
	public int outFeatures() {
		return out;
	}
	
	private void accumulateGrads(Matrix dw, Matrix db) {
		weightGrads = weightGrads.plus(dw);
		biasGrads = biasGrads.plus(db);
	}
	
	
}
