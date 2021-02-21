package hr.fer.nenr.ga.mutations;

import hr.fer.nenr.ga.Chromosome;
import hr.fer.nenr.ga.operators.IMutation;

public class DummyMutation implements IMutation{

	
	@Override
	public Chromosome mutate(Chromosome chromosome) {
		return chromosome;
	}

	
	
}
