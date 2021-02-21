package nenr.fuzzy.coders;

import java.util.function.Function;

import nenr.lab1.fuzzy.IFuzzySet;

public interface IDefuzzifier extends Function<IFuzzySet, Double> {
}
