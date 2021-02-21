package hr.fer.nenr.ga.crossOver;

import hr.fer.nenr.ga.Chromosome;
import hr.fer.nenr.ga.operators.ICrossOver;
import hr.fer.nenr.utils.Utils;

public class DummyCrossOver implements ICrossOver{

	private boolean flipCoin() {
		return Utils.randomDouble(0, 1) >= 0.5;
	}
	
	@Override
	public Chromosome crossOver(Chromosome parent1, Chromosome parent2) {
		return flipCoin() ? parent1 : parent2;
	}

	@Override
	public String name() {
		return "DUMMY";
	}
	
}
