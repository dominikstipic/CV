package hr.fer.zemris.java.tests.dataBase;

import org.junit.Assert;
import org.junit.Test;

import hr.fer.zemris.java.hw05.db.QueryParser;

public class QueryParserTest {

	@Test
	public void forParsing1() {
		QueryParser qp1 = new QueryParser(" jmbag =\"0123456789\" ");
		
		Assert.assertEquals(true, qp1.isDirectQuery());
		Assert.assertEquals("0123456789", qp1.getQueriedJMBAG());
		Assert.assertEquals(1 ,qp1.getQuery().size());
	}
	
	@Test(expected = IllegalStateException.class)
	public void forParsing2() {
		QueryParser qp2 = new QueryParser(" jmbag=\"0123456789\" and lastName>\"J\"");
		
		
		Assert.assertEquals(false, qp2.isDirectQuery());
		Assert.assertEquals(2 ,qp2.getQuery().size());
		
		Assert.assertEquals("0123456789", qp2.getQueriedJMBAG());
		
	}
	
	@Test(expected = IllegalStateException.class)
	public void forParsing3() {
		QueryParser qp = new QueryParser("  jmbag  =  \"0123456789\"  and  lastName   >    \"J\"   and  firstName LIKE  \" so*\"  ");
		
		
		Assert.assertEquals(false, qp.isDirectQuery());
		Assert.assertEquals(3,qp.getQuery().size());
		
		Assert.assertEquals("0123456789", qp.getQueriedJMBAG());
		
	}
	
}
