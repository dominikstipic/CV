package nenr.mutation;

import static nenr.main.GAConfigurator.LOWER;
import static nenr.main.GAConfigurator.MUTATION_PROBABILITY;
import static nenr.main.GAConfigurator.UPPER;
import java.util.List;
import java.util.stream.Collectors;
import nenr.genetic.Chromosome;
import nenr.main.Utils;
import nenr.operators.IMutation;

public class UniformMutation implements IMutation{
	private double PROB = MUTATION_PROBABILITY;
	
	public double randValue() {
		if(!Utils.throwCoin(PROB)) return 0;
		double value = Utils.randDouble(1, LOWER, UPPER)[0];
		return value;
	}
	
	@Override
	public Chromosome mutate(Chromosome chromosome) {
		List<Double> values = chromosome.getOriginal();
		values = values.stream()
				       .map(d -> d + randValue())
				       .collect(Collectors.toList());
		double[] vals = Utils.toPrimitive(values);
		Chromosome mutated = new Chromosome(vals);
		return mutated;
	}
}
