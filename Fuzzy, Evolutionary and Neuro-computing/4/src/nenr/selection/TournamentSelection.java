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

public class TournamentSelection implements ISelection{
	private int K;

	public TournamentSelection(int k) {
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
		return indices;
	}
	
	
	
	private Chromosome tournament(List<Chromosome> population, Comparator<Chromosome> chromosomeComparator) {
		List<Integer> idx = randomIndex(K, population.size());
		if(new HashSet<>(idx).size() != idx.size())  
			throw new IllegalStateException("Selection havent picked unique chromosomes from population");
		List<Chromosome> ranked = idx.stream().map(i -> population.get(i)).collect(Collectors.toList());

		Collections.sort(ranked, chromosomeComparator);
		Chromosome selected = ranked.get(0);            
		return selected;
	}
	
	@Override
	public List<Chromosome> select(int n, Set<Chromosome> population, Comparator<Chromosome> chromosomeComparator) {
		List<Chromosome> matingPool = new ArrayList<>();
		List<Chromosome> P = new ArrayList<>(population);
		while(matingPool.size() != n) {
			Chromosome selected = tournament(P, chromosomeComparator);
			if(matingPool.contains(selected)) continue;
			matingPool.add(selected);
			P.remove(selected);
		}
		return matingPool;
	}
	
	
	
}
