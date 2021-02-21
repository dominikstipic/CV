package nenr.fuzzy.machine;

import java.util.List;
import java.util.function.BiFunction;
import java.util.stream.Collectors;

import nenr.fuzzy.rule.IRule;
import nenr.fuzzy.utils.Utils;
import nenr.lab1.fuzzy.IFuzzySet;

public class InferenceMachine implements IMachine {
	private BiFunction<IFuzzySet, IFuzzySet, IFuzzySet> sNorm;
	private BiFunction<Double, Double, Double> implication;

	public InferenceMachine(BiFunction<IFuzzySet, IFuzzySet, IFuzzySet> sNorm,
			                BiFunction<Double, Double, Double> implication) {
		this.sNorm = sNorm;
		this.implication = implication;
	}

	@Override
	public IFuzzySet apply(List<IFuzzySet> inputs, List<IRule> rules) {
		List<IFuzzySet> outputs = rules.stream().
				                        map(r -> r.apply(inputs, implication)).
				                        collect(Collectors.toList()); 
		//outputs.forEach(o -> Utils.printFuzzy(o, false));
		IFuzzySet y = outputs.get(0);
		outputs.remove(0);
		for (IFuzzySet y_i : outputs) {
			y = sNorm.apply(y, y_i);
		}
		Utils.printFuzzy(y, false);
		return y;
	}

}
