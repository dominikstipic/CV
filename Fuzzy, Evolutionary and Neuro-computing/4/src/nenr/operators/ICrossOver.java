package nenr.operators;

import nenr.genetic.Chromosome;

public interface ICrossOver {
Chromosome crossOver(Chromosome parent1, Chromosome parent2);
}
