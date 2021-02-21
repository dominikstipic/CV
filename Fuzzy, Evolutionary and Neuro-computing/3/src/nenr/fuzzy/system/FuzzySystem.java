package nenr.fuzzy.system;

import java.util.List;
import java.util.stream.Collectors;

import nenr.fuzzy.coders.IDefuzzifier;
import nenr.fuzzy.coders.IFuzzifier;
import nenr.fuzzy.database.FuzzyDatabase;
import nenr.fuzzy.machine.IMachine;
import nenr.fuzzy.rule.IRule;
import nenr.fuzzy.variables.InputVariable;
import nenr.lab1.fuzzy.IFuzzySet;
import nenr.main.Configuration;

public class FuzzySystem implements IFuzzySystem{
	private IDefuzzifier defuzzer;
	private IFuzzifier fuzzer;
	private FuzzyDatabase database;
	private IMachine machine;
	
	public FuzzySystem(FuzzyDatabase database) {
		this.database = database;
		configure();
	}
	
	private void configure() {
		defuzzer = Configuration.getDefuzzer();
		fuzzer = Configuration.getFuzzifier();
		machine = Configuration.getMachine();
	}
	

	@Override
	public Double apply(List<InputVariable> variables) {
		List<IFuzzySet> singletons = variables.stream().
			                         map(v -> fuzzer.apply(v)).
			                         collect(Collectors.toList());
		
//		variables.forEach(v -> System.out.println(v));
//		singletons.forEach(f -> Utils.printFuzzy(f, false));
		
		List<IRule> rules = database.getRules();
		IFuzzySet y = machine.apply(singletons, rules);
		Double output = defuzzer.apply(y);
		return output;
	}

	
}
