package nenr.reproduction;

import java.util.List;

import nenr.genetic.Chromosome;
import nenr.main.Utils;
import nenr.operators.ICrossOver;

public class FloatReproduction implements ICrossOver{

	@Override
	public Chromosome crossOver(Chromosome parent1, Chromosome parent2) {
		List<Double> p1 = parent1.getOriginal();
		List<Double> p2 = parent2.getOriginal();
		double[] child = new double[p1.size()];
		for(int i = 0; i < p1.size(); ++i) {
			double d1 = p1.get(i);
			double d2 = p2.get(i);
			double r;
			if(d1 > d2) {
				r = Utils.randDouble(1, d2, d1)[0];
			}
			else {
				r = Utils.randDouble(1, d1, d2)[0];
			}
			child[i] = r;
		}
		return new Chromosome(child);
	}
	
	
}
