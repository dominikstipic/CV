package hr.fer.zemris.java.tests.dataBase;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import hr.fer.zemris.java.hw05.db.IFilter;
import hr.fer.zemris.java.hw05.db.StudentDatabase;
import hr.fer.zemris.java.hw05.db.StudentRecord;
import hr.fer.zemris.java.hw05.studentDB.StudentDB;

public class StudentDatabaseTest {
	String path = "src/main/resources/database.txt";
	StudentDatabase base;
	{
		try {
			base = new StudentDatabase(StudentDB.DBloader(path));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	@Test
	public void forJMBAG() {
		StudentRecord r0 = base.forJMBAG("0000000001");
		StudentRecord r1 = base.forJMBAG("0000000002");
		StudentRecord r2 = base.forJMBAG("0000000003");
		StudentRecord r3 = base.forJMBAG("0000000004");
		StudentRecord r4 = base.forJMBAG("0000000005");
		
		Assert.assertEquals(new StudentRecord("0000000001", "Akšamović", "Marin", "***") ,r0);
		Assert.assertEquals(new StudentRecord("0000000002", "***", "***", "***") ,r1);
		Assert.assertEquals(new StudentRecord("0000000003", "***", "***", "***") ,r2);
		Assert.assertEquals(new StudentRecord("0000000004", "***", "***", "***") ,r3);
		Assert.assertEquals(new StudentRecord("0000000005", "***", "***", "***") ,r4);
	}
	
	@Test
	public void forFilter() {
		IFilter filterTrue =  ( r -> true);
		IFilter filterFalse =  ( r -> false);
		
		List<StudentRecord> expected1 = base.getRecordList();
		List<StudentRecord> expected2 = new LinkedList<>();
		
		Assert.assertEquals(expected1, base.filter(filterTrue));
		Assert.assertEquals(expected2, base.filter(filterFalse));
	}
	
	
	
	
	
}
