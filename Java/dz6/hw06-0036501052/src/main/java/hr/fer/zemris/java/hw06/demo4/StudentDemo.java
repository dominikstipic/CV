package hr.fer.zemris.java.hw06.demo4;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Demonstrates usage of labmda expressions and usage of <code>Stream</code> API
 * @author Dominik Stipic
 *
 */
public class StudentDemo {

	
	/**
	 * Method which is automaticaly started when program runs.
	 * @param args arguments from command line interface
	 */
	public static void main(String[] args) throws IOException {
		String path = "src/main/resources/studenti.txt";
		List<String> lines = Files.readAllLines(Paths.get(path));
		List<StudentRecord> records = convert(lines);

		System.out.println("broj studenata s 25 ili vise bodova: " + vratiBodovaViseOd25(records));
		
		System.out.println("broj odlikasa: " + broj5(records));
		
		System.out.println("svi odlikasi");
		vratiListuOdlikasa(records).forEach( r-> System.out.println(r.getLastName()));
		System.out.println("******************************");
		
		System.out.println("svi odlikasi sortirano(po bodovima silazno)");
		vratiSortiranuListuOdlikasa(records).forEach(r -> System.out.println(r.examSum()));
		System.out.println("******************************");
		
		System.out.println("svi koji nisu polozili kolegij,sortirano po jmbag uzlazno:");
		vratiPopisNepolozenih(records).forEach(jmbag -> System.out.println(jmbag));
		System.out.println("******************************");
		
		System.out.println("grupirano po ocjenama");
		razvrstajStudentePoOcjenama(records).forEach((grade, list) -> System.out.println(grade + "->" + list));
		System.out.println("******************************");
		
		System.out.println("Svakoj ocjeni pridruzen broj studenata s tom ocjenom");
		vratiBrojStudenataPoOcjenama(records).forEach((grade, num) -> System.out.println(grade + "->" + num));
		System.out.println("******************************");
		
		System.out.println("Prolaz / pad");
		razvrstajProlazPad(records).forEach((bool, studList) -> System.out.println(bool + "->" + studList));
		System.out.println("******************************");
		

		

	}

	/**
	 * Returns number of students with more than 25 points
	 * @param records list of <code>StudentRecord</code>
	 * @return students with more than 25 points
	 */
	public static long vratiBodovaViseOd25(List<StudentRecord> records) {
		return records.stream()
					  .filter(r -> r.getZI() + r.getMI() + r.getLAB() > 25)
					  .count();
	}

	/**
	 * Number of students with grade 5
	 * @param records list of <code>StudentRecord</code> 
	 * @return number of students with grade 5
	 */
	public static long broj5(List<StudentRecord> records) {
		 return records.stream()
				 	   .filter(r -> r.getGrade() == 5)
				 	   .count();
	}

	/**
	 * Removes all the students without grade 5
	 * @param records list of <code>StudentRecord</code>
	 * @return List of <code>StudentRecord</code> with students with grade 5 
	 */
	public static List<StudentRecord> vratiListuOdlikasa(List<StudentRecord> records) {
		return records.stream()
					  .filter(r -> r.getGrade() == 5)
					  .collect(Collectors.toList());
	}

	/**
	 * Removes students without grade 5 and sorts them according the number of points on exams
	 * @param records list of <code>StudentRecord</code>
	 * @return List of <code>StudentRecord</code> with sorted students with grade 5
	 */
	@SuppressWarnings("deprecation")
	public static List<StudentRecord> vratiSortiranuListuOdlikasa(List<StudentRecord> records) {
		return records.stream()
					  .filter(r -> r.getGrade() == 5)
					  .sorted((r1, r2) -> new Double(r2.examSum()).compareTo(new Double(r1.examSum())))     
					  .collect(Collectors.toList());
	}
	
	/**
	 * Leaves students who didn't passed year
	 * @param records list of <code>StudentRecord</code>
	 * @return List of <code>StudentRecord</code> who didnt passsed year
	 */
	public static List<String> vratiPopisNepolozenih(List<StudentRecord> records) {
		return records.stream()
					  .filter(r -> r.getGrade() == 1).map(r -> r.getJmbag())
				      .sorted().collect(Collectors.toList());
	}
	
	/**
	 * Maps the grades with the students who got that grade
	 * @param records list of <code>StudentRecord</code>
	 * @return Map with : key-grade, value-List<StudentRecord>
	 */
	public static Map<Integer,List<StudentRecord>> razvrstajStudentePoOcjenama(List<StudentRecord> records) {
		return records.stream()
					  .collect(Collectors.groupingBy(StudentRecord::getGrade));
	}
	
	/**
	 * Maps the grade with the number of students who got that grade
	 * @param records list of <code>StudentRecord</code>
	 * @return Map with: key - grade, value - number of student with that grade 
	 */
	public static Map<Integer, Integer> vratiBrojStudenataPoOcjenama(List<StudentRecord> records) {
		return records.stream()
					  .collect(Collectors.toMap(StudentRecord::getGrade, 
												   key -> 2, 
												   (value,initialVal) -> value = value + initialVal
												   ));
	}
	
	/**
	 * Maps the student with true = passed year or false = didn't passed year 
	 * @param records list of <code>StudentRecord</code>year
	 * @return Map with : key - boolean value, value - students who passed or didnt passed exam
	 */
	public static Map<Boolean,List<StudentRecord>> razvrstajProlazPad(List<StudentRecord> records) {
		return  records.stream()
					    .collect(Collectors.partitioningBy(r -> r.getGrade() == 1));
	}
	
	
	
	/**
	 * Convetes list of lines with student's info into 
	 * list of <code>StudentRecord</code>
	 * @param lines of text file 
	 * @return List of <code>StudentRecord</code>
	 */
	public static List<StudentRecord> convert(List<String> lines) {
		List<StudentRecord> records = new LinkedList<>();
		lines.forEach(line -> {
			String parts[] = line.split("[ ]+|\\t");
			records.add(new StudentRecord(parts[0], parts[1], parts[2], Double.parseDouble(parts[3]),
					Double.parseDouble(parts[4]), Double.parseDouble(parts[5]), Integer.parseInt(parts[6])));
		});

		return records;
	}

}
