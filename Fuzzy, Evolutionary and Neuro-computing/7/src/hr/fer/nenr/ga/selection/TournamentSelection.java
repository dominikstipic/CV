package hr.fer.nenr.ga.selection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import hr.fer.nenr.ga.Chromosome;
import hr.fer.nenr.ga.operators.ISelection;

public class TournamentSelection implements ISelection{
	private int tournamentSize;
	
	public TournamentSelection(int tournamentSize) {
		this.tournamentSize = tournamentSize;
	}

	private List<Integer> randIndices(int upper) {
		List<Integer> pop = IntStream.range(0, upper).boxed().collect(Collectors.toList());
		Random rand = new Random(System.nanoTime());
		List<Integer> sample = new ArrayList<>();
		for(int i = 0; i < tournamentSize; ++i) {
			int idx = rand.nextInt(pop.size());
			int elem = pop.get(idx);
			pop.remove(idx);
			sample.add(elem);
		}
		Collections.sort(sample);
		return sample;
	}
	
	@Override
	public List<Chromosome> select(List<Chromosome> population) {
		List<Integer> indices = randIndices(population.size());
		List<Chromosome> selected = new ArrayList<>();
		for(int idx : indices) {
			Chromosome chr = population.get(idx);
			selected.add(chr);
		}
		return selected;
	}

	
}
