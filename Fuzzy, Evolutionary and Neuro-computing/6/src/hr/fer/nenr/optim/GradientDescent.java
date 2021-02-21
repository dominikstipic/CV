package hr.fer.nenr.optim;

import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;
import java.util.function.BiFunction;

import Jama.Matrix;
import hr.fer.nenr.blocks.Composite;
import hr.fer.nenr.blocks.ComputationalBlock;
import hr.fer.nenr.models.Parameters;

public class GradientDescent extends Optimizer{
	private double epsilon;
	private BiFunction<Matrix, Matrix, Matrix> update = (W, DW) -> W.minus(DW.times(epsilon));
	
	public GradientDescent(Composite block, double epsilon) {
		super(block);
		this.epsilon = epsilon;
	}
	
	@Override
	public void step() {
		for(ComputationalBlock block : module) {
			if(block.hasParam()) {
				List<Parameters> params = block.getParams();
				Parameters param = params.get(0);
				for(Entry<String, Matrix> e: param.copy()) {
					String key = e.getKey(); Matrix W = e.getValue();
					Matrix dW = param.getGrad(key);
					W = update.apply(W, dW);
					param.addParam(key, W);
				}
				block.setParams(Arrays.asList(param));
			}
		}
	}

	
}
