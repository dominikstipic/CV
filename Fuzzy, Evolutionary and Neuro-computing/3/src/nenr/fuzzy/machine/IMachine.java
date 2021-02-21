package nenr.fuzzy.machine;

import java.util.List;
import java.util.function.BiFunction;

import nenr.fuzzy.rule.IRule;
import nenr.lab1.fuzzy.IFuzzySet;

public interface IMachine extends BiFunction<List<IFuzzySet>, List<IRule>, IFuzzySet>{

}
