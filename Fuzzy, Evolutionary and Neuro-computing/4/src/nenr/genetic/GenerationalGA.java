package nenr.genetic;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;

import nenr.main.GAConfigurator;

public class GenerationalGA extends GA{
	private boolean elitism = GAConfigurator.ELITISM;
	
	public GenerationalGA(Function<Chromosome, Double> optim) {
		super(optim);
		chromosomeComparator = (c1, c2) -> {
			boolean v1 = fitness.apply(c1) > fitness.apply(c2);
			boolean v2 = fitness.apply(c1) < fitness.apply(c2);
			if(v1) return -1;
			else if(v2) return 1;
			else return 0;};
	}
	
	

	@Override
	protected Set<Chromosome> geneticStep() {
		Set<Chromosome> newPopulation = new HashSet<>();
		Chromosome currentBest = getBest();
		double bestValue = fitness.apply(currentBest);
		if(elitism) {
			newPopulation.add(currentBest);
		}
		while(newPopulation.size() != population.size()) {
			List<Chromosome> matingPool = selection.select(2, population, chromosomeComparator);
			Chromosome parent1 = matingPool.get(0);
			Chromosome parent2 = matingPool.get(1);
			Chromosome child = crossOver.crossOver(parent1, parent2);
			child = mutation.mutate(child);
			// Not allowing duplicates
			if(!newPopulation.contains(child)) {
				double childFitness = fitness.apply(child);
				if(childFitness > bestValue) {
					System.out.println("\tNEW HERO: " + child + ", FITNESS: " + childFitness);
					bestValue = childFitness;
					
				}
				newPopulation.add(child);
			}
		}
		return newPopulation;
	}
	
	

}
