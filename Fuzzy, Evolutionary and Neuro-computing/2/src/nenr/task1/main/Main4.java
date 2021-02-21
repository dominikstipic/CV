package nenr.task1.main;

import nenr.task1.Relations;
import nenr.zad1.Domain;
import nenr.zad1.DomainElement;
import nenr.zad1.IDomain;
import nenr.zad2.IFuzzySet;
import nenr.zad2.MutableFuzzySet;

public class Main4 {

	public static void main(String[] args) {
		IDomain u = Domain.intRange(1,5);
		
		IFuzzySet r = new MutableFuzzySet(Domain.combine(u, u))
				      .set(DomainElement.of(1,1), 1)
				      .set(DomainElement.of(2,2), 1)
				      .set(DomainElement.of(3,3), 1)
				      .set(DomainElement.of(4,4), 1)
				      .set(DomainElement.of(1,2), .3)
				      .set(DomainElement.of(2,1), .3)
				      .set(DomainElement.of(2,3), .5)
				      .set(DomainElement.of(3,2), .5)
				      .set(DomainElement.of(3,4), .2)
				      .set(DomainElement.of(4,3), .2);
		
		IFuzzySet r2 = r;
		
		System.out.println("Početna relacija je neizrazita  relacija ekvivalencije? " + 
		                   Relations.isFuzzyEquivalence(r2));
		System.out.println();
		
		for(int i = 1; i <= 3; ++i) {
			r2 = Relations.compositionOfBinaryRelations(r2, r);
			System.out.println("Broj odrđenih kompozicija: " + i + ". Relacija je:");
			for(DomainElement e : r2.getDomain()) {
				System.out.println("mu("+e+")="+r2.getValueAt(e));
			}
			System.out.println("Ova relacija je neizrazita relacija ekvivalencije? " + 
			                  Relations.isFuzzyEquivalence(r2));
			System.out.println();
		}
	}

}
