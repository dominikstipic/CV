package hr.fer.zemris.java.tests.collection;

import java.util.Iterator;
import java.util.NoSuchElementException;

import hr.fer.zemris.java.hw05.collections.SimpleHashtable;
import hr.fer.zemris.java.hw05.collections.SimpleHashtable.TableEntry;

public class SimpleHashtableTest2 {
	
	public static void main(String[] args) {
		SimpleHashtable<String, Integer> examMarks = new SimpleHashtable<>(2);
		// fill data:
		examMarks.put("Ivana", 2);
		examMarks.put("Jura", null);
		examMarks.put("Ante", 2);
		examMarks.put("Jasna", 2);
		examMarks.put("Kristina", 5);
		examMarks.put("Ivana", 5); 
		examMarks.put("Stef", 101);
		examMarks.put("Vlado", null);
		examMarks.put("Vlado", 5);
		
		for(SimpleHashtable.TableEntry<String, Integer> entry : examMarks) {
			for(SimpleHashtable.TableEntry<String, Integer> entry2 : examMarks) {
				System.out.println(entry + " - " + entry2);
			}
		}
		
		try {
			Iterator<SimpleHashtable.TableEntry<String, Integer>> iter = examMarks.iterator();
			System.out.println(iter.next()); //ante
			System.out.println(iter.next()); //ivana
			System.out.println(iter.next()); //jasna	
			System.out.println(iter.next()); //kristina 
			System.out.println(iter.next()); //null	
		} catch (NoSuchElementException e) {
			System.out.println(e);
		}
		Iterator<SimpleHashtable.TableEntry<String, Integer>> iter = examMarks.iterator();
		
		while(iter.hasNext()) {
			TableEntry<String, Integer> currentEntry = iter.next();
			if(currentEntry.getKey().equals("Jasna")) {
				iter.remove();
			}
			if(currentEntry.getKey().equals("Ante")) {
				iter.remove();
			}
		}
		
		System.out.println(examMarks);
		System.out.println(examMarks.size());
		
	}
}
