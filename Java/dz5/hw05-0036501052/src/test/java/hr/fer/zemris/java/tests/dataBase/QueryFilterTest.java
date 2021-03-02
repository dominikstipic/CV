package hr.fer.zemris.java.tests.dataBase;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import hr.fer.zemris.java.hw05.db.QueryFilter;
import hr.fer.zemris.java.hw05.db.QueryParser;
import hr.fer.zemris.java.hw05.db.StudentDatabase;
import hr.fer.zemris.java.hw05.db.StudentRecord;

public class QueryFilterTest {
	public static List<String> DBloader(String path) throws IOException {
		return Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8);
	}

	@Test
	public void forFiltring() throws IOException {
		String path = "src/main/resources/database.txt";
		StudentDatabase db = new StudentDatabase(DBloader(path));
		QueryParser parser = new QueryParser("lastName like \"P*\"");
		
		
		List<StudentRecord> list = db.filter(new QueryFilter(parser.getQuery()));
		
			Assert.assertEquals("0000000042", list.get(0).getJMBAG());
			Assert.assertEquals("Palajić", list.get(0).getLastName());
			Assert.assertEquals("Nikola", list.get(0).getFirstName());
			Assert.assertEquals("3", list.get(0).getFinalGrade());
		
			Assert.assertEquals("0000000043", list.get(1).getJMBAG());
			Assert.assertEquals("Perica", list.get(1).getLastName());
			Assert.assertEquals("Krešimir", list.get(1).getFirstName());
			Assert.assertEquals("4", list.get(1).getFinalGrade());
			
			Assert.assertEquals("0000000044", list.get(2).getJMBAG());
			Assert.assertEquals("Pilat", list.get(2).getLastName());
			Assert.assertEquals("Ivan", list.get(2).getFirstName());
			Assert.assertEquals("5", list.get(2).getFinalGrade());
		
	}
}
