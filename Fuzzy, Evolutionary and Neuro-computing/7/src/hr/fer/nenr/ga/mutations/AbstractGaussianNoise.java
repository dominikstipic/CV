package hr.fer.nenr.ga.mutations;

import java.util.Random;

import hr.fer.nenr.ga.Chromosome;
import hr.fer.nenr.ga.operators.IMutation;
import hr.fer.nenr.utils.Utils;

public abstract class  AbstractGaussianNoise implements IMutation{
	protected double sigma;
	protected double proba;
	
	public AbstractGaussianNoise(double sigma, double proba) {
		this.sigma = sigma;
		this.proba = proba;
	}

	protected double sample() {
		Random rand = new Random(System.nanoTime());
		return sigma * rand.nextGaussian();
	}
	
	@Override
	public Chromosome mutate(Chromosome chromosome) {
		double[] sol = chromosome.copy().solution;
		for(int i = 0; i < sol.length; ++i) {
			if(Utils.randomDouble(0, 1) <= proba) {
				sol[i] = specificMutation(sol[i]);
			}
		}
		Chromosome mutated = new Chromosome(sol);
		return mutated;
	}
	
	abstract double specificMutation(double value);
	
	
}
