package hr.fer.nenr.ga.crossOver;

import hr.fer.nenr.ga.Chromosome;
import hr.fer.nenr.ga.operators.ICrossOver;
import hr.fer.nenr.utils.Utils;

public class WeightCrossOver implements ICrossOver{
	private double prob;
	
	public WeightCrossOver(double prob) {
		this.prob = prob;
	}


	private boolean flipCoin() {
		return Utils.randomDouble(0, 1) <= prob;
	}
	
	
	@Override
	public Chromosome crossOver(Chromosome parent1, Chromosome parent2) {
		double[] xs = parent1.solution;
		double[] ys = parent2.solution;
		if(xs.length != ys.length)
			throw new IllegalArgumentException("Vector sizes should be the same");
		double[] childVec = new double[xs.length];
		for(int i = 0; i < xs.length; ++i) {
			if(flipCoin()) 
				childVec[i] = xs[i];
			else 
				childVec[i] = ys[i];
		}
		return new Chromosome(childVec);
	}

	@Override
	public String name() {
		return "WEIGHT_CROSSOVER";
	}
	
}
