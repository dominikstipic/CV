package hr.fer.zemris.java.tests.dataBase;

import org.junit.Assert;
import org.junit.Test;

import hr.fer.zemris.java.hw05.db.ComparisonOperators;

public class ComparisonOperatorsTest {
	@Test
	public void forLessOperator() {
		Assert.assertEquals(true,ComparisonOperators.LESS.satisfied("Ana", "Jasna"));
		Assert.assertEquals(false,ComparisonOperators.LESS.satisfied("Ana", "Ana"));
		Assert.assertEquals(false,ComparisonOperators.LESS.satisfied("Jasna", "Jasna"));
		Assert.assertEquals(false,ComparisonOperators.LESS.satisfied("Jasna", "Ana"));
		Assert.assertEquals(true,ComparisonOperators.LESS.satisfied("abb", "z"));
		Assert.assertEquals(false,ComparisonOperators.LESS.satisfied("zasdf", "a"));
		Assert.assertEquals(false,ComparisonOperators.LESS.satisfied("a", ""));
	}
	
	@Test
	public void forLessOrEquals() {
		Assert.assertEquals(true,ComparisonOperators.LESS_OR_EQUALS.satisfied("Ana", "Jasna"));
		Assert.assertEquals(true,ComparisonOperators.LESS_OR_EQUALS.satisfied("Ana", "Ana"));
		Assert.assertEquals(true,ComparisonOperators.LESS_OR_EQUALS.satisfied("Jasna", "Jasna"));
		Assert.assertEquals(false,ComparisonOperators.LESS_OR_EQUALS.satisfied("Jasna", "Ana"));
		Assert.assertEquals(true,ComparisonOperators.LESS_OR_EQUALS.satisfied("abb", "z"));
		Assert.assertEquals(false,ComparisonOperators.LESS_OR_EQUALS.satisfied("zasdf", "a"));
		Assert.assertEquals(false,ComparisonOperators.LESS_OR_EQUALS.satisfied("a", ""));
	}
	
	@Test
	public void forGreater() {
		Assert.assertEquals(false,ComparisonOperators.GREATER.satisfied("Ana", "Jasna"));
		Assert.assertEquals(false,ComparisonOperators.GREATER.satisfied("Ana", "Ana"));
		Assert.assertEquals(false,ComparisonOperators.GREATER.satisfied("Jasna", "Jasna"));
		Assert.assertEquals(true,ComparisonOperators.GREATER.satisfied("Jasna", "Ana"));
		Assert.assertEquals(false,ComparisonOperators.GREATER.satisfied("abb", "z"));
		Assert.assertEquals(true,ComparisonOperators.GREATER.satisfied("zasdf", "a"));
		Assert.assertEquals(true,ComparisonOperators.GREATER.satisfied("a", ""));
	}
	
	@Test
	public void forGreaterOrEquals() {
		Assert.assertEquals(false,ComparisonOperators.GREATER_OR_EQUALS.satisfied("Ana", "Jasna"));
		Assert.assertEquals(true,ComparisonOperators.GREATER_OR_EQUALS.satisfied("Ana", "Ana"));
		Assert.assertEquals(true,ComparisonOperators.GREATER_OR_EQUALS.satisfied("Jasna", "Jasna"));
		Assert.assertEquals(true,ComparisonOperators.GREATER_OR_EQUALS.satisfied("Jasna", "Ana"));
		Assert.assertEquals(false,ComparisonOperators.GREATER_OR_EQUALS.satisfied("abb", "z"));
		Assert.assertEquals(true,ComparisonOperators.GREATER_OR_EQUALS.satisfied("zasdf", "a"));
		Assert.assertEquals(true,ComparisonOperators.GREATER_OR_EQUALS.satisfied("a", ""));
	}
	
	@Test
	public void forEquals() {
		Assert.assertEquals(false,ComparisonOperators.EQUALS.satisfied("Ana", "Jasna"));
		Assert.assertEquals(true,ComparisonOperators.EQUALS.satisfied("Ana", "Ana"));
		Assert.assertEquals(true,ComparisonOperators.EQUALS.satisfied("Jasna", "Jasna"));
		Assert.assertEquals(false,ComparisonOperators.EQUALS.satisfied("Jasna", "Ana"));
		Assert.assertEquals(false,ComparisonOperators.EQUALS.satisfied("abb", "z"));
		Assert.assertEquals(false,ComparisonOperators.EQUALS.satisfied("zasdf", "a"));
		Assert.assertEquals(false,ComparisonOperators.EQUALS.satisfied("a", ""));
	}
	
	@Test
	public void forNotEquals() {
		Assert.assertEquals(true,ComparisonOperators.NOT_EQUALS.satisfied("Ana", "Jasna"));
		Assert.assertEquals(false,ComparisonOperators.NOT_EQUALS.satisfied("Ana", "Ana"));
		Assert.assertEquals(false,ComparisonOperators.NOT_EQUALS.satisfied("Jasna", "Jasna"));
		Assert.assertEquals(true,ComparisonOperators.NOT_EQUALS.satisfied("Jasna", "Ana"));
		Assert.assertEquals(true,ComparisonOperators.NOT_EQUALS.satisfied("abb", "z"));
		Assert.assertEquals(true,ComparisonOperators.NOT_EQUALS.satisfied("zasdf", "a"));
		Assert.assertEquals(true,ComparisonOperators.NOT_EQUALS.satisfied("a", ""));
	}
	
	@Test
	public void forLike() {
		Assert.assertEquals(true ,ComparisonOperators.LIKE.satisfied("Zagreb", "Zagreb"));
		Assert.assertEquals(false ,ComparisonOperators.LIKE.satisfied("Zagreb", "Zagrebs"));
		Assert.assertEquals(false ,ComparisonOperators.LIKE.satisfied("Zagreb", "Aba*"));
		Assert.assertEquals(false ,ComparisonOperators.LIKE.satisfied("AAA", "AA*AA")); 
		Assert.assertEquals(false ,ComparisonOperators.LIKE.satisfied("AAAA", "AA*AA")); 
		Assert.assertEquals(true ,ComparisonOperators.LIKE.satisfied("wordgarbagegarbageword", "word*word")); 
		Assert.assertEquals(true ,ComparisonOperators.LIKE.satisfied("garbagegarbagesword", "*word")); 
		Assert.assertEquals(true ,ComparisonOperators.LIKE.satisfied("wordgarbagegarbage", "word*")); 
		Assert.assertEquals(false ,ComparisonOperators.LIKE.satisfied("nenene", "word*")); 
	}
	
	@Test(expected = IllegalArgumentException.class )
	public void forWrongLike() {
		ComparisonOperators.LIKE.satisfied("AAAA", "AA**AA");
	}
}
