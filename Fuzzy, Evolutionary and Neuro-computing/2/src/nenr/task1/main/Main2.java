package nenr.task1.main;

import nenr.task1.Relations;
import nenr.zad1.Domain;
import nenr.zad1.DomainElement;
import nenr.zad2.IFuzzySet;
import nenr.zad2.MutableFuzzySet;

public class Main2 {

	public static void main(String[] args) {

		
		
		IFuzzySet R = new MutableFuzzySet(Domain.combine(Domain.intRange(0, 4), Domain.intRange(0, 4)))
			       .set(DomainElement.of(new int[]{0,0}), 1)
			       .set(DomainElement.of(new int[]{0,1}), .629)
			       .set(DomainElement.of(new int[]{0,2}), .735)
			       .set(DomainElement.of(new int[]{0,3}), .735)
			       .set(DomainElement.of(new int[]{0,4}), .907)
			       .set(DomainElement.of(new int[]{1,0}), .629)
			       .set(DomainElement.of(new int[]{1,1}), 1)
			       .set(DomainElement.of(new int[]{1,2}), .972)
			       .set(DomainElement.of(new int[]{1,3}), .8)
			       .set(DomainElement.of(new int[]{1,4}), .63)
			       .set(DomainElement.of(new int[]{2,0}), .735)
			       .set(DomainElement.of(new int[]{2,1}), .972)
			       .set(DomainElement.of(new int[]{2,2}), 1)
			       .set(DomainElement.of(new int[]{2,3}), .735)
			       .set(DomainElement.of(new int[]{2,4}), .713)
			       .set(DomainElement.of(new int[]{3,0}), .735)
			       .set(DomainElement.of(new int[]{3,1}), .8)
			       .set(DomainElement.of(new int[]{3,2}), .735)
			       .set(DomainElement.of(new int[]{3,3}), 1)
			       .set(DomainElement.of(new int[]{3,4}), .713)
			       .set(DomainElement.of(new int[]{4,0}), .907)
			       .set(DomainElement.of(new int[]{4,1}), .63)
			       .set(DomainElement.of(new int[]{4,2}), .713)
			       .set(DomainElement.of(new int[]{4,3}), .713)
			       .set(DomainElement.of(new int[]{4,4}), 1);
		
		IFuzzySet Q = Relations.compositionOfBinaryRelations(R, R);
		System.out.println("-----");
		for(DomainElement elem : Q.getDomain()) {
			double value = Q.getValueAt(elem);
			System.out.println(elem + ", " + value);
		}
		
	}

}
