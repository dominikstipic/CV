package hr.fer.zemris.java.hw06.demo2;

/**
 * Demonstartes usage of <code>PrimesCollection</code>
 * @author Dominik Stipic
 *
 */
public class PrimesDemo2 {
	
	/**
	 * Method which is automaticaly started when program runs.
	 * @param args arguments from command line interface
	 */
	public static void main(String[] args) {
		PrimesCollection primesCollection = new PrimesCollection(2);
		for(Integer prime : primesCollection){
			for(Integer prime2 : primesCollection){
				System.out.println("Got prime pair: " + prime + ", " + prime2);
			}
		}
	}
	
}
