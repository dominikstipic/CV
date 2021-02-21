package nenr.zad1.main;

import nenr.zad1.Domain;
import nenr.zad1.DomainElement;
import nenr.zad1.IDomain;

public class Main {

	public static void main(String[] args) {
		IDomain d1 = Domain.intRange(0, 5); // {0,1,2,3,4}
		Debug.print(d1, "Elementi domene d1:");
		IDomain d2 = Domain.intRange(0, 3); // {0,1,2}
		Debug.print(d2, "Elementi domene d2:");
		Domain d3 = Domain.combine(d1, d2);
		Debug.print(d3, "Elementi domene d3:");
		
		System.out.println(d3.elementForIndex(0));
		System.out.println(d3.elementForIndex(5));
		System.out.println(d3.elementForIndex(14));
		
		DomainElement de = DomainElement.of(new int[] {4,1});
		System.out.println(d3.IndexOfElement(de));

	}

}
