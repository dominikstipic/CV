package hr.fer.nenr.blocks;

import java.util.Arrays;
import java.util.List;

import Jama.Matrix;
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
	
	private Parameters formParameterPacket() {
		List<String> names = Arrays.asList("W", "b");
		Parameters param = new Parameters(names, Arrays.asList(W, biases));
		param.addGrads(names, Arrays.asList(weightGrads, biasGrads));
		return param;
	}
	
	public void initParams() {
		W = MatrixAdapter.symetricUniform(1, out, in);
		biases = MatrixAdapter.symetricUniform(1, 1, out);
	}
	
	@Override
	public Matrix forward(Matrix inputs) {
		// inputs : 1,N
		memory = inputs;
		Matrix S = W.times(inputs.transpose());
		S = S.transpose();
		S = S.plus(biases);
		return S;
	}

	@Override
	public Matrix backward(Matrix gradients) {
		Matrix dw = gradients.transpose().times(memory); 
		accumulateGrads(dw, gradients);
		Matrix dx = gradients.times(W);
		return dx;
	}

	@Override
	public void cleanGradients() {
		weightGrads = MatrixAdapter.replicate(0, out, in);
		biasGrads = MatrixAdapter.replicate(0, 1, out);
	}

	@Override
	public List<Parameters> getParams() {
		Parameters params = formParameterPacket();
		return Arrays.asList(params);
	}
	
	@Override
	public void setParams(List<Parameters> params) {
		if(params.size() != 1) 
			throw new IllegalArgumentException("Params list should be of size 1");
		Parameters param = params.get(0);
		Matrix w = param.getParam("W");
		Matrix b = param.getParam("b");
		if(!MatrixAdapter.checkDim(w, W) || !MatrixAdapter.checkDim(b, biases))
			throw new IllegalArgumentException("Parameters dimensions doesnt match up");
		W = w; 
		biases = b;
	}

	@Override
	public boolean hasParam() {
		return true;
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
