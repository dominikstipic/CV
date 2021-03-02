package hr.fer.zemris.java.custom.scripting;

import java.util.EmptyStackException;

import org.junit.Assert;
import org.junit.Test;

public class ObjectMultiStackTest {

	@Test
	public void forPushing() {
		ObjectMultistack multistack = new ObjectMultistack();
		
		multistack.push("year", new ValueWrapper(Integer.valueOf(2000)));
		Assert.assertEquals(2000, multistack.peek("year").getValue());
		
		multistack.push("price", new ValueWrapper(200.51));
		Assert.assertEquals(200.51, multistack.peek("price").getValue());
		
		multistack.push("year", new ValueWrapper(Integer.valueOf(1900)));
		Assert.assertEquals(1900, multistack.peek("year").getValue());
		
		multistack.peek("year").setValue(((Integer) multistack.peek("year").getValue()).intValue() + 50);
		Assert.assertEquals(1950, multistack.peek("year").getValue());
		
		multistack.pop("year");
		Assert.assertEquals(2000, multistack.peek("year").getValue());
		
		multistack.peek("year").add("5");
		Assert.assertEquals(2005, multistack.peek("year").getValue());
		multistack.peek("year").add(5);
		Assert.assertEquals(2010, multistack.peek("year").getValue());
		multistack.peek("year").add(5.0);
		Assert.assertEquals(2015.0, multistack.peek("year").getValue());
	}
	
	@Test
	public void forPoping() {
		ObjectMultistack multistack = new ObjectMultistack();
		
		multistack.push("year", new ValueWrapper(Integer.valueOf(2000)));
		multistack.push("year", new ValueWrapper(1.1));
		multistack.push("year", new ValueWrapper(1.81));
		multistack.push("year", new ValueWrapper(101));
		multistack.push("year", new ValueWrapper(12.1));
		
		ValueWrapper w = multistack.pop("year");
		Assert.assertEquals(12.1, w.getValue());
		
		w = multistack.pop("year");
		Assert.assertEquals(101,w.getValue());
		
		w = multistack.pop("year");
		Assert.assertEquals(1.81, w.getValue());
		
		w = multistack.pop("year");
		Assert.assertEquals(1.1, w.getValue());
		
		w = multistack.pop("year");
		Assert.assertEquals(2000, w.getValue());
	}
	
	@Test(expected=EmptyStackException.class)
	public void forEmptyStack() {
		ObjectMultistack multistack = new ObjectMultistack();
		
		multistack.push("year", new ValueWrapper(Integer.valueOf(2000)));
		multistack.push("year", new ValueWrapper(1.1));
		multistack.push("year", new ValueWrapper(1.81));
		multistack.push("year", new ValueWrapper(101));
		multistack.push("year", new ValueWrapper(12.1));
		
		multistack.pop("year");
		multistack.pop("year");
		multistack.pop("year");
		multistack.pop("year");
		multistack.pop("year");
		
		multistack.pop("year"); //throws
	}

}
