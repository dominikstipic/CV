package hr.fer.zemris.java.custom.scripting;

import java.util.concurrent.LinkedBlockingDeque;

import org.junit.Assert;
import org.junit.Test;

public class ValueWraperTest {

	@Test
	public void forAdding() {
		ValueWrapper v1 = new ValueWrapper(null);
		ValueWrapper v2 = new ValueWrapper(null);
		v1.add(v2.getValue());
		Assert.assertEquals(0, v1.getValue());
		
		v1 = new ValueWrapper("1.2E1");
		v2 = new ValueWrapper(Integer.valueOf(1));
		v1.add(v2.getValue());
		Assert.assertEquals(13.0, v1.getValue());
		
		v1 = new ValueWrapper("12");
		v2 = new ValueWrapper(Integer.valueOf(1));
		v1.add(v2.getValue());
		Assert.assertEquals(13, v1.getValue());
		
		v1 = new ValueWrapper(null);
		v2 = new ValueWrapper(2.45);
		v1.add(v2.getValue());
		Assert.assertEquals(2.45, v1.getValue());
		
		v1 = new ValueWrapper("345.2E-1");
		v2 = new ValueWrapper(2.45);
		v1.add(v2.getValue());
		Assert.assertEquals(34.52+2.45, v1.getValue());
		
		v1 = new ValueWrapper(-1);
		v2 = new ValueWrapper(2.45);
		v1.add(v2.getValue());
		Assert.assertEquals(-1+2.45, v1.getValue());
		
		v1 = new ValueWrapper("24");
		v2 = new ValueWrapper("234.3");
		v1.add(v2.getValue());
		Assert.assertEquals(24+234.3, v1.getValue());
		
		v1 = new ValueWrapper("24");
		v2 = new ValueWrapper("-234.3");
		v1.add(v2.getValue());
		Assert.assertEquals(24-234.3, v1.getValue());
		
		v1 = new ValueWrapper("24E-2");
		v2 = new ValueWrapper("1.2E100");
		v1.add(v2.getValue());
		Assert.assertEquals(24E-2 + 1.2E100, v1.getValue());
		
	}
	
	@Test
	public void forSubtraction() {
		ValueWrapper v1 = new ValueWrapper(null);
		ValueWrapper v2 = new ValueWrapper(null);
		v1.subtract(v2.getValue());
		Assert.assertEquals(0, v1.getValue());
		
		v1 = new ValueWrapper("1.2E1");
		v2 = new ValueWrapper(Integer.valueOf(1));
		v1.subtract(v2.getValue());
		Assert.assertEquals(1.2E1 - 1, v1.getValue());
		
		v1 = new ValueWrapper("12");
		v2 = new ValueWrapper(Integer.valueOf(1));
		v1.subtract(v2.getValue());
		Assert.assertEquals(11, v1.getValue());
		
		v1 = new ValueWrapper(null);
		v2 = new ValueWrapper(2.45);
		v1.subtract(v2.getValue());
		Assert.assertEquals(-2.45, v1.getValue());
		
		v1 = new ValueWrapper("345.2E-1");
		v2 = new ValueWrapper(2.45);
		v1.subtract(v2.getValue());
		Assert.assertEquals(34.52-2.45, v1.getValue());
		
		v1 = new ValueWrapper(-1);
		v2 = new ValueWrapper(2.45);
		v1.subtract(v2.getValue());
		Assert.assertEquals(-1-2.45, v1.getValue());
		
		v1 = new ValueWrapper("24");
		v2 = new ValueWrapper("234.3");
		v1.subtract(v2.getValue());
		Assert.assertEquals(24-234.3, v1.getValue());
		
		v1 = new ValueWrapper("24");
		v2 = new ValueWrapper("-234.3");
		v1.subtract(v2.getValue());
		Assert.assertEquals(24+234.3, v1.getValue());
		
		v1 = new ValueWrapper("24E-2");
		v2 = new ValueWrapper("1.2E100");
		v1.subtract(v2.getValue());
		Assert.assertEquals(24E-2 - 1.2E100, v1.getValue());
	}
	
	@Test
	public void forMultiplication() {
		ValueWrapper v1 = new ValueWrapper(null);
		ValueWrapper v2 = new ValueWrapper(null);
		v1.subtract(v2.getValue());
		Assert.assertEquals(0, v1.getValue());
		
		v1 = new ValueWrapper("1.2E1");
		v2 = new ValueWrapper(Integer.valueOf(1));
		v1.multiply(v2.getValue());
		Assert.assertEquals(1.2E1*1, v1.getValue());
		
		v1 = new ValueWrapper("12");
		v2 = new ValueWrapper(Integer.valueOf(1));
		v1.multiply(v2.getValue());
		Assert.assertEquals(12, v1.getValue());
		
		v1 = new ValueWrapper(null);
		v2 = new ValueWrapper(2.45);
		v1.multiply(v2.getValue());
		Assert.assertEquals(0.0, v1.getValue());
		
		v1 = new ValueWrapper("345.2E-1");
		v2 = new ValueWrapper(2.45);
		v1.multiply(v2.getValue());
		Assert.assertEquals(34.52*2.45, v1.getValue());
		
		v1 = new ValueWrapper(-1);
		v2 = new ValueWrapper(2.45);
		v1.multiply(v2.getValue());
		Assert.assertEquals(-1*2.45, v1.getValue());
		
		v1 = new ValueWrapper("24");
		v2 = new ValueWrapper("234.3");
		v1.multiply(v2.getValue());
		Assert.assertEquals(24*234.3, v1.getValue());
		
		v1 = new ValueWrapper("24");
		v2 = new ValueWrapper("-234.3");
		v1.multiply(v2.getValue());
		Assert.assertEquals(-24*234.3, v1.getValue());
		
		v1 = new ValueWrapper("24E-2");
		v2 = new ValueWrapper("1.2E100");
		v1.multiply(v2.getValue());
		Assert.assertEquals(24E-2*1.2E100, v1.getValue());
		
		v1 = new ValueWrapper(null);
		v2 = new ValueWrapper("0");
		v1.multiply(v2.getValue());
		Assert.assertEquals(0, v1.getValue());
	}
	
	@Test
	public void forDivison() {
		ValueWrapper v1 = new ValueWrapper("6.567");
		ValueWrapper v2 = new ValueWrapper(32);
		v1.divide(v2.getValue());
		Assert.assertEquals(6.567 / 32, v1.getValue());
		
		v1 = new ValueWrapper("1.2E1");
		v2 = new ValueWrapper(Integer.valueOf(1));
		v1.divide(v2.getValue());
		Assert.assertEquals(1.2E1/1, v1.getValue());
		
		v1 = new ValueWrapper("12");
		v2 = new ValueWrapper(Integer.valueOf(1));
		v1.divide(v2.getValue());
		Assert.assertEquals(12, v1.getValue());
		
		v1 = new ValueWrapper(null);
		v2 = new ValueWrapper(2.45);
		v1.divide(v2.getValue());
		Assert.assertEquals(0.0, v1.getValue());
		
		v1 = new ValueWrapper("345.2E-1");
		v2 = new ValueWrapper(2.45);
		v1.divide(v2.getValue());
		Assert.assertEquals(34.52/2.45, v1.getValue());
		
		v1 = new ValueWrapper(-1);
		v2 = new ValueWrapper(2.45);
		v1.divide(v2.getValue());
		Assert.assertEquals(-1 / 2.45, v1.getValue());
		
		v1 = new ValueWrapper("24");
		v2 = new ValueWrapper("234.3");
		v1.divide(v2.getValue());
		Assert.assertEquals(24 / 234.3, v1.getValue());
		
		v1 = new ValueWrapper("24");
		v2 = new ValueWrapper("-234.3");
		v1.divide(v2.getValue());
		Assert.assertEquals(-24 / 234.3, v1.getValue());
		
		v1 = new ValueWrapper("24E-2");
		v2 = new ValueWrapper("1.2E100");
		v1.divide(v2.getValue());
		Assert.assertEquals(24E-2 / 1.2E100, v1.getValue());
		
		v1 = new ValueWrapper(null);
		v2 = new ValueWrapper("8");
		v1.divide(v2.getValue());
		Assert.assertEquals(0, v1.getValue());
		
		v1 = new ValueWrapper(5);
		v2 = new ValueWrapper("2");
		v1.divide(v2.getValue());
		Assert.assertEquals(2, v1.getValue());
		
		v1 = new ValueWrapper(5);
		v2 = new ValueWrapper("2.0");
		v1.divide(v2.getValue());
		Assert.assertEquals(2.5, v1.getValue());
	}
	
	@Test
	public void forComparsion() {
		ValueWrapper v1 = new ValueWrapper("6.567");
		ValueWrapper v2 = new ValueWrapper(32);
		int comp = v1.numCompare(v2.getValue());
		Assert.assertTrue(comp < 0);
		
		v1 = new ValueWrapper("1.2E1");
		v2 = new ValueWrapper(Integer.valueOf(1));
		comp = v1.numCompare(v2.getValue());
		Assert.assertTrue(comp > 0);
		
		v1 = new ValueWrapper("12");
		v2 = new ValueWrapper(Integer.valueOf(1));
	    comp = v1.numCompare(v2.getValue());
		Assert.assertTrue(comp > 0);
		
		v1 = new ValueWrapper(null);
		v2 = new ValueWrapper(2.45);
	    comp = v1.numCompare(v2.getValue());
		Assert.assertTrue(comp < 0);
		
		v1 = new ValueWrapper("345.2E-1");
		v2 = new ValueWrapper(2.45);
	    comp = v1.numCompare(v2.getValue());
		Assert.assertTrue(comp > 0);
		
		v1 = new ValueWrapper(-1);
		v2 = new ValueWrapper(2.45);
		comp = v1.numCompare(v2.getValue());
		Assert.assertTrue(comp < 0);
		
		v1 = new ValueWrapper("24");
		v2 = new ValueWrapper("234.3");
		comp = v1.numCompare(v2.getValue());
		Assert.assertTrue(comp < 0);
		
		v1 = new ValueWrapper("24");
		v2 = new ValueWrapper("-234.3");
		comp = v1.numCompare(v2.getValue());
		Assert.assertTrue(comp > 0);
		
		v1 = new ValueWrapper("24E-2");
		v2 = new ValueWrapper("1.2E100");
		comp = v1.numCompare(v2.getValue());
		Assert.assertTrue(comp < 0);
		
		v1 = new ValueWrapper(null);
		v2 = new ValueWrapper("8");
		comp = v1.numCompare(v2.getValue());
		Assert.assertTrue(comp < 0);
		
		v1 = new ValueWrapper(2);
		v2 = new ValueWrapper("2");
		comp = v1.numCompare(v2.getValue());
		Assert.assertTrue(comp == 0);
		
		v1 = new ValueWrapper(2.2);
		v2 = new ValueWrapper("2.2");
		comp = v1.numCompare(v2.getValue());
		Assert.assertTrue(comp == 0);
		
		v1 = new ValueWrapper("12E1");
		v2 = new ValueWrapper("12E1");
		comp = v1.numCompare(v2.getValue());
		Assert.assertTrue(comp == 0);
		
		v1 = new ValueWrapper("12.3E-1");
		v2 = new ValueWrapper("12.3E-1");
		comp = v1.numCompare(v2.getValue());
		Assert.assertTrue(comp == 0);
		
		v1 = new ValueWrapper("12");
		v2 = new ValueWrapper("12.1");
		comp = v1.numCompare(v2.getValue());
		Assert.assertTrue(comp < 0);
		
		v1 = new ValueWrapper("0");
		v2 = new ValueWrapper("0.0001");
		comp = v1.numCompare(v2.getValue());
		Assert.assertTrue(comp < 0);
		
		v1 = new ValueWrapper(0.0);
		v2 = new ValueWrapper("0.0001");
		comp = v1.numCompare(v2.getValue());
		Assert.assertTrue(comp < 0);
	}
	
	@Test(expected = ArithmeticException.class)
	public void forDividingWithZero() {
		ValueWrapper v1 = new ValueWrapper(12);
		ValueWrapper v2 = new ValueWrapper("0.000");
		v1.divide(v2.getValue());
	}
	
	@Test(expected = ArithmeticException.class)
	public void forDividingWithNull() {
		ValueWrapper v1 = new ValueWrapper(1.2);
		ValueWrapper v2 = new ValueWrapper(null);
		v1.divide(v2.getValue());
	}
	
	@Test(expected = RuntimeException.class)
	public void forIllegalArguments2() {
		ValueWrapper v1 = new ValueWrapper(12);
		ValueWrapper v2 = new ValueWrapper(new LinkedBlockingDeque<>());
		v1.divide(v2.getValue());
	}
}
