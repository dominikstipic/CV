package hr.fer.nenr.ga.mutations;

import java.util.List;

import hr.fer.nenr.ga.Chromosome;
import hr.fer.nenr.ga.operators.IMutation;
import hr.fer.nenr.utils.LinAlg;
import hr.fer.nenr.utils.Utils;

public class MutationComposition implements IMutation{
	private List<IMutation> composition;
	private double[] probabilities;

	public MutationComposition(List<IMutation> composition, double ...ts) {
		if(ts.length != composition.size())
			throw new IllegalArgumentException("Desirability list and Mutation list must have the same lenght");
		this.composition = composition;
		this.probabilities = calcProbs(ts);
		
	}
	
	public MutationComposition(List<IMutation> composition) {
		this(composition, LinAlg.rep(1.0, composition.size()));
	}
	
	private double[] calcProbs(double[] ts) {
		double[] probabilities = new double[ts.length];
		double sum = LinAlg.sum(ts);
		for(int i = 0; i < ts.length; ++i)
			probabilities[i] = ts[i]/sum;
		return probabilities;
		
	}
	
	@Override
	public Chromosome mutate(Chromosome chromosome) {
		int N = composition.size();
		for(int i = 0; i < N; ++i) {
			double random = Utils.randomDouble(0, 1);
			if(random <= probabilities[i]) {
				chromosome = composition.get(i).mutate(chromosome);
			}
		}
		return chromosome;
	}

	
	
}
