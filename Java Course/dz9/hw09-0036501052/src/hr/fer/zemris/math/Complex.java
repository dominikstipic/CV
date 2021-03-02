package hr.fer.zemris.math;
import static java.lang.Math.PI;
import static java.lang.Math.atan2;
import static java.lang.Math.cos;
import static java.lang.Math.sin;
import static java.lang.Math.sqrt;
import static java.lang.Math.pow;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;



/**
 * Models Complex number and its apporpriate mathemtical operations.
 * Complex number is defined with 2 parametars:read and imaginary numbers
 * @author Dominik Stipić
 *
 */
public class Complex {
	/**
	 * real part
	 */
	private final double re;
	/**
	 * iamginary part
	 */
	private final double im;
	/**
	 * 0+0i
	 */
	public static final Complex ZERO = new Complex(0,0);
	/**
	 * 1+0i
	 */
	public static final Complex ONE = new Complex(1,0);
	/**
	 * -1+0i
	 */
	public static final Complex ONE_NEG = new Complex(-1,0);
	/**
	 * i
	 */
	public static final Complex IM = new Complex(0,1);
	/**
	 * -1
	 */
	public static final Complex IM_NEG = new Complex(0,-1);
	
	/**
	 * Creates default zero complex:0+0i
	 */
	public Complex() {
		re = 0;
		im = 0;
	}
	
	/**
	 * Creates new complex with given parametars
	 * @param re real part 
	 * @param im imaginary part
	 */
	public Complex(double re, double im) {
		this.re = re;
		this.im = im;
	}
	
	/**
	 * returns module of this complex number
	 * @return module 
	 */
	public double module() {
		return sqrt(re*re + im*im);
	}
	
	/**
	 * Multiplies two complex numbers
	 * @param c other complex
	 * @return reuslt of multiplication
	 */
	public Complex multiply(Complex c) {
		Objects.requireNonNull(c, "Complex Number cannot be null");
		double x = re*c.re - im*c.im;
		double y = im*c.re + re*c.im;
		return new Complex(x, y);
	}
	
	/**
	 * Divides two complex numbers 
	 * @param c other complex
	 * @return result of division
	 */
	public Complex divide(Complex c) {
		Objects.requireNonNull(c, "Complex Number cannot be null");
		double tmp = c.re*c.re + c.im*c.im;
		if(tmp == 0) {
			throw new ArithmeticException("Dividing with zero is undefined");
		}
		double newRe = (double) (re*c.re + im*c.im) / tmp;
		double newIm = (double) (im*c.re - re*c.im) / tmp;
		return new Complex(newRe, newIm);
	}
	
	/**
	 * Addes two complex numbers 
	 * @param c other complex
	 * @return result of addition
	 */
	public Complex add(Complex c) {
		return new Complex(re + c.re, im + c.im);
	}
	
	/**
	 * Subtract two complex numbers 
	 * @param c other complex
	 * @return result of subtraction
	 */
	public Complex sub(Complex c) {
		return new Complex(re-c.re, im-c.im);
	}
	
	/**
	 * negates thos complex
	 * @return negated complex
	 */
	public Complex negate() {
		return new Complex(-1*re,-1*im);
	}
	
	/**
	 * calualltes n-th power of this complex
	 * @param n exponeent
	 * @return powered complex
	 */
	public Complex power(int n) {
		if(n < 0) {
			throw new IllegalArgumentException("n must be non-negative integer");
		}
		List<Double> list = getPolarCoordiantes();
		double r = pow(list.get(0),n);
		double w = n*list.get(1);
		
		double x = r * cos(w);
		double y = r * sin(w);
		return new Complex(x, y);
	}
	
	/**
	 * finds all n-roots 
	 * @param n roots
	 * @return list of roots
	 */
	public List<Complex> root(int n) {
		List<Complex> roots = new LinkedList<>();
		List<Double> list = getPolarCoordiantes();
		double complexMagnitude = Math.pow(list.get(0), (double)1/n);
		double angle = list.get(1);
		
		for (int k = 0; k < n; ++k) {
			double complexAngle = (double)( (angle + 2*k*PI) / n);
			double x = complexMagnitude * cos(complexAngle);
			double y = complexMagnitude * sin(complexAngle);
			roots.add(new Complex(x, y));
		}
		
		return roots;
	}
	
	/**
	 * calculates polar coordinates
	 * @return list of polar components
	 */
	private List<Double> getPolarCoordiantes(){
		double angle = atan2(im, re);
		double r = sqrt(re*re + im*im);
		return List.of(r, angle);
	}

	@Override
	public String toString() {
		if(im >= 0) {
			return String.format("%.6f + %.6fi", re,im); 
		}
		return String.format("%.6f - %.6fi",re, -1*im);
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		long temp;
		temp = Double.doubleToLongBits(im);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(re);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Complex other = (Complex) obj;
		if (Double.doubleToLongBits(im) != Double.doubleToLongBits(other.im))
			return false;
		if (Double.doubleToLongBits(re) != Double.doubleToLongBits(other.re))
			return false;
		return true;
	}

	/**
	 * gets real part
	 * @return real part
	 */
	public double getRe() {
		return re;
	}

	/**
	 * gets imag part
	 * @return imaginary part
	 */
	public double getIm() {
		return im;
	}
	
	
	
}
