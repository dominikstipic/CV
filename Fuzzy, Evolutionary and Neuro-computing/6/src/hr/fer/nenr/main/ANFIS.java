package hr.fer.nenr.main;

import java.util.List;

import Jama.Matrix;
import hr.fer.nenr.blocks.ComputationalBlock;
import hr.fer.nenr.blocks.FuzzyLayer;
import hr.fer.nenr.blocks.Linear;
import hr.fer.nenr.blocks.NormalizationLayer;
import hr.fer.nenr.blocks.ParametrizedBlock;
import hr.fer.nenr.blocks.ProductWindow;
import hr.fer.nenr.blocks.Summer;
import hr.fer.nenr.utils.MatrixAdapter;

public class ANFIS extends ParametrizedBlock{
	private ParametrizedBlock antecedentBlock;
	private Linear linear;
	private Summer summer;
	
	private Matrix F;
	private Matrix W;
	
	public ANFIS(int rules) {
		super(build(rules));
		antecedentBlock = (ParametrizedBlock) children.get(0);
		linear = (Linear) children.get(1);
		summer = (Summer) children.get(2);
		//product = new Product(2);
	}
	
	public static ComputationalBlock[] build(int rules) {
		FuzzyLayer layer1 = new FuzzyLayer(2, 2*rules);
		ProductWindow layer2 = new ProductWindow(2*rules, rules);
		NormalizationLayer layer3 = new NormalizationLayer(rules);
		
		ParametrizedBlock antecedentBlock = new ParametrizedBlock(layer1, layer2, layer3);
		Linear linear = new Linear(2, rules);
		Summer summer = new Summer(rules);
		return new ComputationalBlock[] {antecedentBlock, linear, summer};
	}

	@Override
	public Matrix forward(Matrix inputs) {
		W = antecedentBlock.forward(inputs);
		F = linear.forward(inputs);
		Matrix H = MatrixAdapter.timesElementwise(W, F);
		Matrix output = summer.forward(H);
		return output;
	}

	@Override
	public Matrix backward(Matrix gradients) {
		gradients = summer.backward(gradients);
		Matrix dF = MatrixAdapter.timesElementwise(gradients, W);
		Matrix dW = MatrixAdapter.timesElementwise(gradients, F);
		
		Matrix linearGrads = linear.backward(dF);
		Matrix antGrads = antecedentBlock.backward(dW);
		Matrix grads = MatrixAdapter.stack(antGrads, linearGrads,false);
		return grads;
	}
	
	public List<Double> evaluateRules(Matrix input){
		W = antecedentBlock.forward(input);
		List<Double> activations = MatrixAdapter.fromVec(W);
		return activations;
	}
	
	
	

}
