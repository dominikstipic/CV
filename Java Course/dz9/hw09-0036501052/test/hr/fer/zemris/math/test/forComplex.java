package hr.fer.zemris.math.test;

import java.util.List;

import org.junit.Test;

import hr.fer.zemris.math.Complex;
import junit.framework.Assert;

public class forComplex {
	Complex c0 = new Complex(1,5);
	Complex c1 = new Complex(0.8, 1.3);
	
	@Test
	public void forModule() {
		Assert.assertEquals(Math.sqrt(26), c0.module());
		Assert.assertEquals(Math.sqrt(0.8*0.8+1.3*1.3), c1.module());
	}
	
	@Test
	public void forMultiplication() {
		Assert.assertEquals(new Complex(-5.7,5.3), c0.multiply(c1));
	}
	
	@Test
	public void forDivision() {
		double r =c0.divide(c1).getRe();
		double i =c0.divide(c1).getIm();
		Assert.assertEquals(3.13305, r,0000.1);
		Assert.assertEquals(1.15880, i,0000.1);
	}
	
	@Test
	public void forPower() {
		double re = c0.power(3).getRe();
		double im = c0.power(3).getIm();
		Assert.assertEquals(-74,re,0.00001);
		Assert.assertEquals(-110,im,0.00001);
	}
	
	@Test
	public void forRoot() {
		List<Complex> l = c0.root(2);
		double re = c0.root(2).get(0).getRe();
		double im = c0.root(2).get(0).getIm();
		Assert.assertEquals(1.7462,re,0.0001);
		Assert.assertEquals(1.4316,im,0.0001);

		 re = c0.root(2).get(1).getRe();
		 im = c0.root(2).get(1).getIm();
		 Assert.assertEquals(-1.7463,re,0.0001);
		 Assert.assertEquals(-1.4316,im,0.0001);
	}
}
