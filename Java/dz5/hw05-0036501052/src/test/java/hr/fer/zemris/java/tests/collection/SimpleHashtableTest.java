package hr.fer.zemris.java.tests.collection;

import hr.fer.zemris.java.hw05.collections.SimpleHashtable;

public class SimpleHashtableTest {

	public static void main(String[] args) {
		// create collection:
		SimpleHashtable<String, Integer> examMarks = new SimpleHashtable<>(2);
		// fill data:
		
		examMarks.put("Ivana", 2);
		examMarks.put("Jura", null);
		examMarks.put("Ante", 2);
		examMarks.put("Jasna", 2);
		examMarks.put("Stef", 101);
		examMarks.put("Vlado", null);
		examMarks.put("Kristina", 5);
		examMarks.put("Ivana", 5); // overwrites old grade for Ivana
		examMarks.put("Vlado", 5);
		
		System.out.println("------------------------");
		for(SimpleHashtable.TableEntry<String, Integer> entry : examMarks) {
				System.out.println(entry + " - " + entry);
		}
		
		
		System.out.println(examMarks);
		System.out.println("size: " + examMarks.size());
		
		// query collection:
		Integer kristinaGrade = examMarks.get("Kristina");
		System.out.println("Kristina's exam grade is: " + kristinaGrade); // writes: 5
		System.out.println("Ante exam grade is: " + examMarks.get("Ante")); 
		System.out.println("Ivana exam grade is: " + examMarks.get("Ivana")); 
		System.out.println("Vlado exam grade is: " + examMarks.get("Vlado")); 
		
		System.out.println("Number of stored pairs: " + examMarks.size()); 
		
		System.out.println("contains 5: " + examMarks.containsValue(5));
		System.out.println("contains 2: " + examMarks.containsValue(2));
		System.out.println("contains 6: " + examMarks.containsValue(6));
		System.out.println("contains null: " + examMarks.containsValue(null));
		
		
		System.out.println("contains Kristina: " + examMarks.containsKey("Kristina"));
		examMarks.remove("Kristina");
		System.out.println("contains Kristina: " + examMarks.containsKey("Kristina"));
		
		System.out.println("contains Vlado: " + examMarks.containsKey("Vlado"));
		examMarks.remove("Vlado");
		System.out.println("contains Vlado: " + examMarks.containsKey("Vlado"));
		
		System.out.println("contains Ivana: " + examMarks.containsKey("Ivana"));
		examMarks.remove("Ivana");
		System.out.println("contains Ivana: " + examMarks.containsKey("Ivana"));
		
		System.out.println(examMarks);
		System.out.println("Number of stored pairs: " + examMarks.size()); 
		examMarks.remove("Ivana");
		examMarks.remove("Ante");
		examMarks.remove("Kristina");
		System.out.println(examMarks);
		System.out.println("Number of stored pairs: " + examMarks.size()); 
		System.out.println("------------------------");
		for(SimpleHashtable.TableEntry<String, Integer> entry : examMarks) {
				System.out.println(entry );
		}
	}
}
