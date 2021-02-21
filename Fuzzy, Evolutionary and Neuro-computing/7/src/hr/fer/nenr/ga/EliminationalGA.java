package hr.fer.nenr.ga;

import java.util.List;

import hr.fer.nenr.dataset.IDataset;
import hr.fer.nenr.models.Data;
import hr.fer.nenr.models.NeuralNet;

public class EliminationalGA extends GA{

	public EliminationalGA(NeuralNet net, IDataset<Data> dataset) {
		super(net, dataset);
	}

	private Chromosome geneticOperators(Chromosome p1, Chromosome p2) {
		Chromosome child = crossOver.crossOver(p1.copy(), p2.copy());
		child = mutation.mutate(child);
 		return child;
	}

	@Override
	protected void geneticStep() {
		List<Chromosome> matingPool = selection.select(population);
		int poolSize = matingPool.size();
		Chromosome first = matingPool.get(0);
		Chromosome sec = matingPool.get(1);
		Chromosome worst = matingPool.get(poolSize-1);
		Chromosome child = geneticOperators(first, sec);
		population.remove(worst);
		rankedInsert(child);
		
		populationFitness -= calcFitness(worst);
		populationFitness += calcFitness(child);
	}

	

}