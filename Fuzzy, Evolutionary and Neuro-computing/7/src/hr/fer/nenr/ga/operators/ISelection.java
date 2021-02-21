package hr.fer.nenr.ga.operators;

import java.util.List;

import hr.fer.nenr.ga.Chromosome;

public interface ISelection {
	List<Chromosome> select(List<Chromosome> population);
}
