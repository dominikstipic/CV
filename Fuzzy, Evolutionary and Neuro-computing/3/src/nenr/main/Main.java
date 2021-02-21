
package nenr.main;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Scanner;

import nenr.fuzzy.system.FuzzySystemFactory;
import nenr.fuzzy.system.IFuzzySystem;
import nenr.fuzzy.variables.InputVariable;



public class Main {
	
    public static void main(String[] args) throws IOException {
    	System.out.println("Bok");
    	BufferedReader input = new BufferedReader(new InputStreamReader(System.in));
	    int L=0,D=0,LK=0,DK=0,V=0,S=0;
	    String line = null;
		
	    IFuzzySystem fsAkcel = FuzzySystemFactory.akcelFuzzy();
	    IFuzzySystem fsKormilo = FuzzySystemFactory.kormiloFuzzy();
	    while(true){
			if((line = input.readLine())!=null){
				if(line.charAt(0)=='K') break;
				try(Scanner s = new Scanner(line)){
					L = s.nextInt(); D = s.nextInt(); LK = s.nextInt(); DK = s.nextInt(); V = s.nextInt(); S = s.nextInt();
				}
	        }
			List<InputVariable> vars = InputVariable.toVars(L,D,LK,DK,V,S);
			Double akcel = fsAkcel.apply(vars);
			Double kormilo = fsKormilo.apply(vars);
	        System.out.println(akcel + " " + kormilo);
	        System.out.flush(); 
	   }
    }

}

