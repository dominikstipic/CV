package nenr.fuzzy.database;

import java.util.ArrayList;
import java.util.List;

import nenr.fuzzy.rule.IRule;
import nenr.fuzzy.rule.Rule;
import nenr.lab1.fuzzy.IFuzzySet;

public abstract class DatabaseFactory {
	
	protected abstract List<List<IFuzzySet>> getAntecenents();
	
	protected abstract List<IFuzzySet> getConsequences();
	
	public FuzzyDatabase createDatabase() {
		List<IRule> rules = getRules();
		FuzzyDatabase database = new FuzzyDatabase(rules);
		return database;
	}
	
	public List<IRule> getRules() {
		List<List<IFuzzySet>> antecedentes = getAntecenents();
		List<IFuzzySet> consequences = getConsequences();
		
		List<IRule> rules = new ArrayList<>();
		for(int i = 0; i < antecedentes.size(); ++i) {
			IRule rule = new Rule(antecedentes.get(i), consequences.get(i));
			rules.add(rule);
		}
		return rules;
	}
	
}
