package hr.fer.nenr.optim;

import Jama.Matrix;
import hr.fer.nenr.blocks.Component;
import hr.fer.nenr.interfaces.ComputationalBlock;
import hr.fer.nenr.models.Parameters;

public class GradientDescent extends Optimizer{
	private double epsilon;
	
	public GradientDescent(Component block, double epsilon) {
		super(block);
		this.epsilon = epsilon;
	}
	
	@Override
	public void step() {
		for(ComputationalBlock block : module) {
			if(block.hasParam()) {
				Matrix[] grads = block.getGrads();
				Parameters params = block.getParams();
				Matrix gradW = grads[0]; Matrix gradB = grads[1];
				Matrix W = params.W; Matrix b = params.b;
				
				Matrix deltaW = gradW.times(epsilon);
				Matrix deltaB = gradB.times(epsilon);
				
				W = W.plus(deltaW);
				b = b.plus(deltaB);
				
				block.setParams(new Parameters(W, b));
			}
		}
	}

	
}
