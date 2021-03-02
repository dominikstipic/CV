package hr.fer.zemris.java.hw06.demo2;


/**
 * Demonstartes usage of <code>PrimesCollection</code>
 * @author Dominik Stipic
 *
 */
public class PrimesDemo1 {
	
	/**
	 * Method which is automaticaly started when program runs.
	 * @param args arguments from command line interface
	 */
	public static void main(String[] args) {
		PrimesCollection primesCollection = new PrimesCollection(100);
		for(Integer prime:primesCollection){
			System.out.println("Got prime: " + prime);
		}
	}
}
