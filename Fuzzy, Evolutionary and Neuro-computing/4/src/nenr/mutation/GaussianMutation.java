package nenr.mutation;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import nenr.genetic.Chromosome;
import nenr.main.GAConfigurator;
import nenr.main.Utils;
import nenr.operators.IMutation;

public class GaussianMutation implements IMutation{
	private double PROB = GAConfigurator.MUTATION_PROBABILITY;
	
	private double gaussianNoise() {
		if(!Utils.throwCoin(PROB)) {
			return 0;
		}
		long time = System.nanoTime();
		double value = new Random(time).nextGaussian();
		return value;
	}
	
	@Override
	public Chromosome mutate(Chromosome chromosome) {
		List<Double> values = chromosome.getOriginal();
		values = values.stream()
				       .map(d -> d + gaussianNoise())
				       .collect(Collectors.toList());
		double[] vals = Utils.toPrimitive(values);
		Chromosome mutated = new Chromosome(vals);
		return mutated;
	}

	
}
