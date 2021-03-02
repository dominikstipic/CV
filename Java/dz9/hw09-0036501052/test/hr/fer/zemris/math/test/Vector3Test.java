package hr.fer.zemris.math.test;


import org.junit.Test;

import hr.fer.zemris.math.Vector3;
import junit.framework.Assert;


public class Vector3Test {
	Vector3 a = new Vector3(1,1,1);
	Vector3 b = new Vector3(1,2,3);
	
	
	@Test
	public void forNorm() {
		Assert.assertEquals(Math.sqrt(3), a.norm());
		Assert.assertEquals(Math.sqrt(14), b.norm());
	}
	
	@Test
	public void forDot() {
		Assert.assertEquals(6., a.dot(b));
	}
	
	@Test
	public void forCross() {
		Assert.assertEquals(new Vector3(1.,-2.,1.), a.cross(b));
	}
	
	@Test
	public void forScale() {
		Assert.assertEquals(new Vector3(-2,-2,-2), a.scale(-2));
		Assert.assertEquals(new Vector3(-2,-4,-6), b.scale(-2));
	}
	
	@Test
	public void forCosAngle() {
		Assert.assertEquals(0.9258200997725514, a.cosAngle(b),0.00001);
	}
}