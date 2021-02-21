package hr.fer.nenr.ga.operators;

import hr.fer.nenr.ga.Chromosome;

public interface ICrossOver {
Chromosome crossOver(Chromosome parent1, Chromosome parent2);
String name();
}
