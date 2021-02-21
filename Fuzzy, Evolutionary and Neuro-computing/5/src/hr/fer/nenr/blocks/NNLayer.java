package hr.fer.nenr.blocks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import hr.fer.nenr.interfaces.ComputationalBlock;
import hr.fer.nenr.utils.VectorUtils;

public class NNLayer extends Component{
	private ComputationalBlock activationBlock;
	
	public NNLayer(int inFeatures, int outFeatures) {
		super(buildComponents(inFeatures, outFeatures));
		in = inFeatures; 
		out = outFeatures;
		activationBlock = new Sigmoid(outFeatures);
	}
	
	@Override
	public List<Double> forward(List<Double> inputs) {
		List<Double> result = new ArrayList<>();
		for(ComputationalBlock block : children) {
			Double out = block.forward(inputs).get(0);
			result.add(out);
		}
		result = activationBlock.forward(result);
		return result;
	}
	

	public List<Double> backward(List<Double> gradients) {
		List<ComputationalBlock> reversed = new ArrayList<>(children);
		Collections.reverse(reversed);
		gradients = activationBlock.backward(gradients);
		System.out.println(gradients);
		List<Double> results = new ArrayList<>();
		for(int i = 0; i < reversed.size(); ++i) {
			ComputationalBlock block = reversed.get(i);
			List<Double> g = Arrays.asList(gradients.get(i));
			List<Double> dx = block.backward(g);
			Double s = VectorUtils.sum(dx).get(0);
			results.add(s);
		}
		return results;
	}

	private static List<ComputationalBlock> buildComponents(int in, int out){
		List<ComputationalBlock> children = new ArrayList<>();
		for(int i = 0; i < out; ++i) {
			ComputationalBlock s = new Summator(in);
			children.add(s);
		}
		return children;
	}
	

	@Override
	public void cleanGradients() {
		children.forEach(c -> c.cleanGradients());
	}

	@Override
	public Object getParams() {
		List<List<Double>> matrix = new ArrayList<>();
		for(int i = 0; i < matrix.size(); ++i) {
			if(children.get(i).hasParam()) {
				List<Double> vector = (List<Double>) children.get(i).getParams();
				matrix.add(vector);
			}
		}
		return matrix;
	}

	@Override
	public void setParams(Object params) {
		List<List<Double>> matrix = (List<List<Double>>) params;
		for(int i = 0; i < matrix.size(); ++i) {
			List<Double> vector = matrix.get(i);
			if(children.get(i).hasParam()) {
				children.get(i).setParams(vector);
			}
		}
	}

	@Override
	public boolean hasParam() {
		return true;
	}

	@Override
	public Object getGrads() {
		List<List<Double>> matrix = new ArrayList<>();
		for(ComputationalBlock block : children) {
			if(block.hasParam()) {
				List<Double> vector = (List<Double>) block.getGrads();
				matrix.add(vector);
			}
		}
		return matrix;
	}
	
	
	
}
