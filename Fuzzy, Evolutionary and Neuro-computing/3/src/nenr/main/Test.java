
package nenr.main;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import nenr.fuzzy.system.FuzzySystemFactory;
import nenr.fuzzy.system.IFuzzySystem;
import nenr.fuzzy.variables.InputVariable;



public class Test {
	
	public static int get() {
		int x = new Random(System.nanoTime()).nextInt(10);
		return x;
	}
	
    public static void main(String[] args) throws IOException {
	    int L=get(),D=get(),LK=get(),DK=get(),V=get(),S=1;
		System.out.println(Arrays.asList(L ,D, LK, DK, V, S).toString());
	    IFuzzySystem fsAkcel = FuzzySystemFactory.akcelFuzzy();
	    IFuzzySystem fsKormilo = FuzzySystemFactory.kormiloFuzzy();
		List<InputVariable> vars = InputVariable.toVars(L,D,LK,DK,V,S);
		double akcel = fsAkcel.apply(vars);
		double kormilo = fsKormilo.apply(vars);
        System.out.println(akcel + " " + kormilo);
	   }
    
    }


