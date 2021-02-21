package nenr.selection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

import nenr.genetic.Chromosome;
import nenr.operators.ISelection;

public class Torurnament implements ISelection{
	private int K;

	public Torurnament(int k) {
		K = k;
	}

	private List<Integer> randomIndex(int n, int upper) {
		List<Integer> indices = new ArrayList<>();
		while(indices.size() != n) {
			long time = System.currentTimeMillis();
			int idx = new Random(time).nextInt(upper);
			if(indices.contains(idx)) continue;
			indices.add(idx);
		}
		if(new HashSet<>(indices).size() != indices.size())  
			throw new IllegalStateException("Selection havent picked unique chromosomes from population");
		
		return indices;
	}
	
	@Override
	public List<Chromosome> select(int n, Set<Chromosome> population, Comparator<Chromosome> chromosomeComparator) {
		List<Chromosome> populationList = population.stream().collect(Collectors.toList());
		List<Integer> idx = randomIndex(K, population.size());
		List<Chromosome> picked = idx.stream().map(i -> populationList.get(i)).collect(Collectors.toList());
		Collections.sort(picked, chromosomeComparator);
		return picked;
	}
	
	
	
}
