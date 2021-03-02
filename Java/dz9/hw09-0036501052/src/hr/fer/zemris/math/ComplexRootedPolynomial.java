package hr.fer.zemris.math;

import java.util.LinkedList;
import java.util.List;
import java.util.Objects;

/**
 * Represents complex rooted polynomial form.
 * Provides method for mathematical operation and to transformation to 
 * cannonic polynomial form
 * @author Dominik Stipić
 *
 */
public class ComplexRootedPolynomial {
	/**
	 * complex roots 
	 */
	List<Complex> roots = new LinkedList<>();
	
	/**
	 * Creates rooted polyinomial
	 * @param roots roots of polynomail
	 */
	public ComplexRootedPolynomial(Complex ...roots) {
		Objects.requireNonNull(roots, "Complex cannot be null");
		for(Complex c : roots) {
			this.roots.add(c);
		}
	}
	
	/**
	 * applying some indenpendent variable to polynome
	 * @param z indenpendent varibale
	 * @return value 
	 */
	public Complex apply(Complex z) {
		List<Complex> added = new LinkedList<>();
		roots.forEach(c -> added.add(z.add(c.negate())));
		Complex mul = new Complex(1, 0);
		for(Complex c:added) {
			mul = mul.multiply(c);
		}
		return mul;
	}

	@Override
	public String toString() {
		StringBuilder b = new StringBuilder();
		for(Complex c : roots) {
			b.append("[z - ( "+c+" )] * ");
		}
		b.deleteCharAt(b.length()-2);
		return b.toString().trim();
	}
	
	/**
	 * transforamtion to standarnd polynomail form
	 * @return cannonical polynome form
	 */
	public ComplexPolynomial toComplexPolynom() {
		List<ComplexPolynomial> list = new LinkedList<>();
		for(Complex c:roots) {
			list.add(new ComplexPolynomial(Complex.ONE, c.negate()));
		}
		ComplexPolynomial p = null;
		for(ComplexPolynomial polynome: list) {
			if(p == null) {
				p = polynome;
				continue;
			}
			p = p.multiply(polynome);
		}
		return p;
	}
	
	/**
	 * Find index of cloosest root for given number which is not within treshold
	 * @param z complex
	 * @param treshold 
	 * @return index
	 */
	public int indexOfClosestRootFor(Complex z, double treshold) {
		Objects.requireNonNull(z, "Complex number cannot be null");
		if(treshold < 0) {
			throw new IllegalArgumentException("treshold cannot be negative number");
		}
		
		double diff = treshold;
		int index = -1;
		for(Complex c : roots) {
			double x = z.sub(c).module();
			if(x  < diff) {
				diff = x;
				index = roots.indexOf(c);
			}
		}
		return index;
	}
	
}
