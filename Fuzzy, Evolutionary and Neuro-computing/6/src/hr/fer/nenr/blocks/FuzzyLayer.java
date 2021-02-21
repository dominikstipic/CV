package hr.fer.nenr.blocks;

import java.util.Arrays;
import java.util.List;

import Jama.Matrix;
import hr.fer.nenr.models.Parameters;
import hr.fer.nenr.utils.MatrixAdapter;

public class FuzzyLayer implements ComputationalBlock{
	private Matrix inputs;
	private Matrix outputs;
	private Matrix A;
	private Matrix B;
	private Matrix Agrads; //1, out
	private Matrix Bgrads; // 1, out
	private int in; 
	private int out;
	private int numOfRules;
	private Sigmoid activation; 
	
	public FuzzyLayer(int inFeatures, int outFeatures) {
		in = inFeatures; 
		out = outFeatures;
		numOfRules = out/in;
		activation = new Sigmoid(out);
		cleanGradients();
		initParams();
	}
	
	public void initParams() {
		A = MatrixAdapter.symetricUniform(1, 1, out);
		B = MatrixAdapter.symetricUniform(1, 1, out);
	}
	
	@Override
	public void cleanGradients() {
		Agrads = MatrixAdapter.replicate(0, 1, out);
		Bgrads = MatrixAdapter.replicate(0, 1, out);
	}
	
	public static Matrix expandInput(Matrix inputs, int rules) {
		Matrix result = inputs.copy();
		for(int i = 1; i < rules; ++i) {
			result = MatrixAdapter.stack(result, inputs, true);
		}
		return result;
	}
	
	@Override
	public Matrix forward(Matrix inputs) {
		// inputs : 1,N
		int N = inputs.getColumnDimension();
		if(N != in) 
			throw new IllegalArgumentException(String.format("in = %s, N = %s!!", in, N));
		inputs = expandInput(inputs, numOfRules);
		this.inputs  = inputs.copy();
		Matrix linear = inputs.minus(A);
		linear = MatrixAdapter.timesElementwise(linear, B);
		Matrix result = activation.forward(linear.times(-1));
		this.outputs = result.copy();
		return result;
	}

	private static Matrix sigmoidDerivitive(Matrix outputs) {
		int N = outputs.getColumnDimension();
		Matrix e = MatrixAdapter.replicate(1.0, 1, N);
		Matrix dSigmoid = MatrixAdapter.timesElementwise(e.minus(outputs), outputs);
		return dSigmoid;
	}
	
	private Matrix inputGrads(Matrix gradients, Matrix dA) {
		Matrix a = dA.times(-1);
		Matrix[] kernels = MatrixAdapter.split(a, in);
		Matrix[] grads = MatrixAdapter.split(gradients, in);
		Matrix dInput = MatrixAdapter.replicate(0.0, 1, in);
		for(int i = 0; i < in; ++i) {
			Matrix K = kernels[i];
			Matrix G = grads[i];
			double val = K.times(G.transpose()).get(0, 0);
			dInput.set(0, i, val);
		}
		return dInput;
	}
	
	@Override
	public Matrix backward(Matrix gradients) {
		Matrix dSigmoid = sigmoidDerivitive(outputs);
		Matrix dA = MatrixAdapter.timesElementwise(dSigmoid, B);
		Matrix dB = MatrixAdapter.timesElementwise(inputs.minus(A), dSigmoid).times(-1);
		dA = MatrixAdapter.timesElementwise(dA, gradients);
		dB = MatrixAdapter.timesElementwise(dB, gradients);
		accumulateGrads(dA, dB);
		
		Matrix dx = inputGrads(gradients, dA);
		
		return dx;
	}

	private Parameters formParameterPacket() {
		Parameters param = new Parameters(Arrays.asList("A", "B"), Arrays.asList(A, B));
		param.addGrads(Arrays.asList("A", "B"), Arrays.asList(Agrads, Bgrads));
		return param;
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
		Matrix a = param.getParam("A");
		Matrix b = param.getParam("B");
		if(!MatrixAdapter.checkDim(a, A) || !MatrixAdapter.checkDim(b, B))
			throw new IllegalArgumentException("Parameters dimensions doesnt match up");
		B = b;
		A = a;
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
	
	private void accumulateGrads(Matrix dA, Matrix dB) {
		Agrads = Agrads.plus(dA);
		Bgrads = Bgrads.plus(dB);
	}
}
