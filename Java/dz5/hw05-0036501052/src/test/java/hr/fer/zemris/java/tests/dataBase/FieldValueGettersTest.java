package hr.fer.zemris.java.tests.dataBase;

import org.junit.Assert;
import org.junit.Test;

import hr.fer.zemris.java.hw05.db.FieldValueGetters;
import hr.fer.zemris.java.hw05.db.StudentRecord;

public class FieldValueGettersTest {

	@Test
	public void forFieldValueGetters1() {
		StudentRecord r = new StudentRecord("0", "Marko", "Markic", "***");
		Assert.assertEquals("Marko" ,FieldValueGetters.FIRST_NAME.get(r)); 
		Assert.assertEquals("Markic" ,FieldValueGetters.LAST_NAME.get(r)); 
		Assert.assertEquals("0" ,FieldValueGetters.JMBAG.get(r)); 
	}
	
	@Test
	public void forFieldValueGetters2() {
		StudentRecord r = new StudentRecord("11", "s", "a", "***");
		Assert.assertEquals("s" ,FieldValueGetters.FIRST_NAME.get(r)); 
		Assert.assertEquals("a" ,FieldValueGetters.LAST_NAME.get(r)); 
		Assert.assertEquals("11" ,FieldValueGetters.JMBAG.get(r)); 
	}
}
