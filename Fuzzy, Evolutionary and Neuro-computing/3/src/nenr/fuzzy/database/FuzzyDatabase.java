package nenr.fuzzy.database;

import java.util.Iterator;
import java.util.List;

import nenr.fuzzy.rule.IRule;

public class FuzzyDatabase implements Iterable<IRule>{
	private List<IRule> rules;

	public FuzzyDatabase(List<IRule> rules) {
		this.rules = rules;
	}

	@Override
	public Iterator<IRule> iterator() {
		return rules.iterator();
	}
	
	public IRule get(int i) {
		return rules.get(i);
	}
	
	public List<IRule> getRules(){
		return rules;
	}
}
