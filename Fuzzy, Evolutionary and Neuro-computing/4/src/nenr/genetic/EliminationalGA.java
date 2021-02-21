package nenr.genetic;

import java.util.List;
import java.util.Set;
import java.util.function.Function;

public class EliminationalGA extends GA{

	public EliminationalGA(Function<Chromosome, Double> optim) {
		super(optim);
		chromosomeComparator = (c1, c2) -> {
			boolean v1 = fitness.apply(c1) > fitness.apply(c2);
			boolean v2 = fitness.apply(c1) < fitness.apply(c2);
			if(v1) return -1;
			else if(v2) return 1;
			else return 0;};
	}
	
	private Chromosome reproduce(Chromosome p1, Chromosome p2) {
		Chromosome child = crossOver.crossOver(p1, p2);
		child = mutation.mutate(child);
		return child;
	}
	
	@Override
	protected Set<Chromosome> geneticStep() {
		Chromosome currentBest = getBest();
		double bestValue = fitness.apply(currentBest);
		List<Chromosome> matingPool = selection.select(0, population, chromosomeComparator);
		Chromosome first = matingPool.get(0);
		Chromosome sec = matingPool.get(1);
		Chromosome worst = matingPool.get(2);
		Chromosome child = reproduce(first, sec);
		population.remove(worst);
		population.add(child);
		
		double childFitness = fitness.apply(child);
		if(childFitness > bestValue) {
			System.out.println("\tNEW HERO: " + child + ", FITNESS: " + childFitness);
		}
		return population;
	}

}
