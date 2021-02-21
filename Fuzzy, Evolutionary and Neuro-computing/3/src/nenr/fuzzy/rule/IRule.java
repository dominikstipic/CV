package nenr.fuzzy.rule;

import java.util.List;
import java.util.function.BiFunction;

import nenr.lab1.fuzzy.IFuzzySet;

public interface IRule extends BiFunction<List<IFuzzySet>, BiFunction<Double, Double, Double>, IFuzzySet>{}
