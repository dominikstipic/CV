package nenr.fuzzy.coders;

import nenr.fuzzy.utils.Utils;
import nenr.lab1.domain.IDomain;
import nenr.lab1.domain.SimpleDomain;
import nenr.lab1.fuzzy.IFuzzySet;

public class FuzzerFactory {
	public static final IFuzzifier SINGLETON = input -> {
		IDomain domain = input.getDomain();
		int value = input.getValue();
		if(!SimpleDomain.isSimple(domain)) {
			throw new IllegalArgumentException("Provided domain must be simple domain");
		}
		int n = domain.getCardinality();
		int first =  domain.elementForIndex(0).getValues()[0];
		int last  = domain.elementForIndex(n-1).getValues()[0];
		if(value < first || value > last) {
			throw new IllegalArgumentException("Inputs' value isn't into specified domain");
		}
		IDomain singletonDomain = SimpleDomain.intRange(value, value);
		IFuzzySet singleton = Utils.fuzzyGenerator(singletonDomain, 1.0);
		return singleton;
		};
		

}
