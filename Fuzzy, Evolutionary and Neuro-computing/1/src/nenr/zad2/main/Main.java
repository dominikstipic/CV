package nenr.zad2.main;

import nenr.zad1.Domain;
import nenr.zad1.DomainElement;
import nenr.zad1.IDomain;
import nenr.zad2.CalculatedFuzzySet;
import nenr.zad2.IFuzzySet;
import nenr.zad2.IIntUnaryFunction;
import nenr.zad2.MutableFuzzySet;
import nenr.zad2.StandardFuzzySets;

public class Main {

	public static void main(String[] args) {
		IDomain d = Domain.intRange(0, 11); // {0,1,...,10}
		IFuzzySet set1 = new MutableFuzzySet(d)
		.set(DomainElement.of(0), 1.0)
		.set(DomainElement.of(1), 0.8)
		.set(DomainElement.of(2), 0.6)
		.set(DomainElement.of(3), 0.4)
		.set(DomainElement.of(4), 0.2);
		Debug.print(set1, "Set1:");
		
		IDomain d2 = Domain.intRange(-5, 6); // {-5,-4,...,4,5}
		IIntUnaryFunction function = StandardFuzzySets.lambdaFunction(d2.IndexOfElement(DomainElement.of(-4)),
																	  d2.IndexOfElement(DomainElement.of( 0)),
																	  d2.IndexOfElement(DomainElement.of( 4)));
		
		IFuzzySet set2 = new CalculatedFuzzySet(d2, function);
		Debug.print(set2, "Set2:");
	}

}
