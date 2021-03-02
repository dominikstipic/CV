package hr.fer.zemris.math;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

/**
 * represents standarnd polynome form.
 * Contains Mathematial operaions for efficient handling with this polynome type.
 * @author Dominik Stipić
 *
 */
public class ComplexPolynomial {
	/**
	 * constants
	 */
	List<Complex> poly = new LinkedList<>();

	/**
	 * Creates polynome
	 * @param factors
	 */
	public ComplexPolynomial(Complex ...factors) {
		for(Complex c : factors) poly.add(c);
	}

	/**
	 * returns order of this polynome
	 * @return order
	 */
	public short order() {
		return (short) ((short)poly.size()-1);
	}

	/**
	 * Multiplies two polynomes
	 * @param p second polynome
	 * @return rsult of multiplication
	 */
	public ComplexPolynomial multiply(ComplexPolynomial p) {
		Map <Integer,Complex> map = new TreeMap<>();
		
		for(int i = 0; i < poly.size(); ++i) {
			int order1 = order() - i;
			for(int j = 0; j < p.poly.size(); ++j) {
				int order2 = p.order() - j;
				Complex factor = poly.get(i).multiply(p.poly.get(j)); 
				int resOrder = order1 + order2;
				map.merge(resOrder, factor, (v1,v2)-> {return v1.add(v2);});
			}
		}
		List<Complex> list= map.entrySet().stream().map(e -> e.getValue()).collect(Collectors.toList());
		Collections.reverse(list);
		return new ComplexPolynomial(list.toArray(new Complex[0]));
		}

	/**
	 * derives given polynome
	 * @return derivation of p
	 */
	public ComplexPolynomial derive() {
		List<Complex> der = new LinkedList<>();
		for(int i = 0; i < poly.size() - 1; ++i) {
			int index = (poly.size()-1) - i;
			der.add(poly.get(i).multiply(new Complex(index, 0)));
		}
		return new ComplexPolynomial(der.toArray(new Complex[0]));
	}
	
	/**
	 * Applies some varible to polynome
	 * @param z indenpendent varible
	 * @return value
	 */
	public Complex apply(Complex z) {
		Complex c = Complex.ZERO;
		for(int i = 0; i < poly.size(); ++i) {
			int index = (poly.size()-1) - i;
			Complex part = z.power(index).multiply(poly.get(i));
			c = c.add(part);
		}
		return c;
	}
	
	@Override
	public String toString() {
		StringBuilder b = new StringBuilder();
		
		for(int i = 0; i <poly.size(); ++i) {
			int index = (poly.size()-1) - i;
			b.append("( "+ poly.get(i) +" ) z^" + index + " + ");
		}
		b.deleteCharAt(b.length()-1);
		b.deleteCharAt(b.length()-1);
		return b.toString();
	}
}
