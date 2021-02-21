package nenr.task1.main;

import nenr.task1.Relations;
import nenr.zad1.Domain;
import nenr.zad1.DomainElement;
import nenr.zad1.IDomain;
import nenr.zad2.IFuzzySet;
import nenr.zad2.MutableFuzzySet;

public class Main3 {

	public static void main(String[] args) {
		IDomain u1 = Domain.intRange(1, 5);
		IDomain u2 = Domain.intRange(1, 4);
		IDomain u3 = Domain.intRange(1, 5);
		
		IFuzzySet r1 = new MutableFuzzySet(Domain.combine(u1, u2))
				.set(DomainElement.of(1,1), .3)
				.set(DomainElement.of(1,2), 1)
				.set(DomainElement.of(3,3), .5)
				.set(DomainElement.of(4,3), .5);
		
		IFuzzySet r2 = new MutableFuzzySet(Domain.combine(u2, u3))
				.set(DomainElement.of(1,1), 1)
				.set(DomainElement.of(2,1), .5)
				.set(DomainElement.of(2,2), .7)
				.set(DomainElement.of(3,3), 1)
				.set(DomainElement.of(3,4), .4);
		
		IFuzzySet r1r2 = Relations.compositionOfBinaryRelations(r1, r2);
		
		for(DomainElement e : r1r2.getDomain()) {
			System.out.println("mu("+e+")=" + r1r2.getValueAt(e));
		}
		
	}

}
