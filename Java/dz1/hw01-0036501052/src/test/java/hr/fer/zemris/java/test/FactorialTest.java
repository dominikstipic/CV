package hr.fer.zemris.java.test;


import org.junit.Assert;
import org.junit.Test;

import hr.fer.zemris.java.hw01.Factorial;

public class FactorialTest {

	@Test (expected = IllegalArgumentException.class) 
	public void forExceptionError() {
		Factorial.factorial(-1);
		Factorial.factorial(-20);
		Factorial.factorial(21);
		Factorial.factorial(31);
		Factorial.factorial(-14);
		
	}
	
	@Test
	public void forDefinedInterval() {
		for(int i = Factorial.MIN_INPUT;i < Factorial.MAX_INPUT;++i ) {
			Assert.assertEquals(true, Factorial.isInInterval(i));
		}
	}
	
	@Test 
	public void forUndefinedInterval() {		
		for(int i = 1;i < Factorial.MAX_INPUT;++i ) {
			int negative = i*(-1);			//undefined negative number
			Assert.assertEquals(false, Factorial.isInInterval(negative));
		}
	}
}
