package nenr.fuzzy.rule;

import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

import nenr.lab1.domain.DomainElement;
import nenr.lab1.domain.IDomain;
import nenr.lab1.fuzzy.IFuzzySet;
import nenr.lab1.fuzzy.MutableFuzzySet;

public class Rule implements IRule{
	private int numberOfVars;
	private List<IFuzzySet> antecedents;
	private IFuzzySet consequence;
	
	public Rule(List<IFuzzySet> antecedents, IFuzzySet consequence) {
		this.consequence = consequence;
		this.antecedents = antecedents;
		numberOfVars = antecedents.size();
	}
	
	private Double calcAntecedent(List<IFuzzySet> localInputs, BiFunction<Double, Double, Double> implicationFunction) {
		DomainElement activeElement = getActiveElement(localInputs);
		Function<Integer, DomainElement> domainIdx = i -> DomainElement.of(activeElement.getComponentValue(i));
		
		Double miValue = null;
		for(int i = 0; i < numberOfVars; ++i) {
			IFuzzySet Ai = antecedents.get(i);
			DomainElement xi = domainIdx.apply(i);
			double miAi = Ai.getValueAt(xi);
			if(miValue == null) miValue = miAi;
			else miValue = implicationFunction.apply(miValue, miAi);
			
		}
		return miValue;
	}
	
	private DomainElement getActiveElement(List<IFuzzySet> singletons) {
		List<Integer> singletonElements = singletons.stream().map(f -> f.getDomain().elementForIndex(0).getValues()[0]).collect(Collectors.toList());
		DomainElement activeElement = DomainElement.fromList(singletonElements);
		return activeElement;
	}
	
	
	public IFuzzySet apply(List<IFuzzySet> localInputs, BiFunction<Double, Double, Double> implicationFunction) {
		// localInputs : Singletons
		if(localInputs.size() != numberOfVars) {
			throw new IllegalArgumentException("The number of variables doesn't match with number of inputs");
		}
		double antValue = calcAntecedent(localInputs, implicationFunction);
		IDomain domain = consequence.getDomain();
		MutableFuzzySet result = new MutableFuzzySet(domain);
		for(DomainElement elem : domain) {
			double mi = consequence.getValueAt(elem);
			double newValue = implicationFunction.apply(mi, antValue);
			result.set(elem, newValue);
		}
		return result;
	}
	

	public List<IFuzzySet> getAntecedens() {
		return antecedents;
	}

	public IFuzzySet getConsequence() {
		return consequence;
	}
}
