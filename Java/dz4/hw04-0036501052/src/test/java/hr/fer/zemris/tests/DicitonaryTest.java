package hr.fer.zemris.tests;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

import hr.fer.zemris.java.custom.collections.Dictionary;

public class DicitonaryTest {
	String nums[] = {"one","two","three","four","five","six","seven","eight","nine","ten"};
	String letters[]  = {"a","b","c","d","e","f","g","h","i","j"};
	
	@Test
	public void forPuttingAndGetting(){
		Dictionary dic = new Dictionary();

		fillUp(dic);
		
		for(int i = 0;i < nums.length ; ++i) {
			assertEquals(i+1, (int) dic.get(nums[i]));
		}
	}
	
	@Test
	public void forRewritting(){
		Dictionary dic = new Dictionary();
		
		fillUp(dic);
		
		for(int i = 0;i < nums.length ; ++i) {
			dic.put(nums[i], letters[i]);
		}
		
		for(int i = 0;i < nums.length ; ++i) {
			assertEquals(letters[i],dic.get(nums[i]));
		}
		
	}
	
	@Test
	public void forSize(){
		Dictionary dic = new Dictionary();
		
		fillUp(dic);
		
		assertEquals(10, dic.size());
		
		for(int i = 0;i < nums.length ; ++i) {
			dic.put(nums[i], letters[i]);
		}
		
		assertEquals(10, dic.size());
		
		dic.clear();
		
		assertEquals(0, dic.size());
		
	}
	
	@Test
	public void forAddingNull(){
		Dictionary dic = new Dictionary();
		
		dic.put(nums[0], 1);
		dic.put(nums[1], 2);
		dic.put("this is null", null);
		dic.put(nums[2], 3);
		
		
		assertEquals(1, dic.get(nums[0]));
		assertEquals(2, dic.get(nums[1]));
		assertEquals(3, dic.get(nums[2]));
		assertEquals(null, dic.get("this is null"));
		
		
	}
	
	@Test(expected = NullPointerException.class)
	public void forKeyIsNull() {
		Dictionary dic = new Dictionary();
		
		dic.put(null, 1);
		
		
	}
	
	
	
	public void fillUp(Dictionary dic) {
		for(int i = 0; i < nums.length ; ++i) {
			dic.put(nums[i],i+1);
		}
		
	}
	
}
