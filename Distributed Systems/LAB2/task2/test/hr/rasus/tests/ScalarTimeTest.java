package hr.rasus.tests;

import java.util.List;

import org.junit.Test;

import hr.fer.rasus.interfaces.LogicalClock;
import hr.fer.rasus.time.ScalarLogicalClock;

public class ScalarTimeTest {
	private LogicalClock p1 = new ScalarLogicalClock(6);
	private LogicalClock p2 = new ScalarLogicalClock(5);
	private LogicalClock p3 = new ScalarLogicalClock(10);
	
	private void clock() {
		p1.increment();
		p2.increment();
		p3.increment();
	}
	
	private void send(LogicalClock source, LogicalClock target) {
		source.increment();
		target.update(source.get());
	}
	
	private void print(LogicalClock ...clocks) {
		for(LogicalClock c : clocks) System.out.println(c);
		System.out.println("------");
	}
	
	private void print() {
		List<LogicalClock> clocks = List.of(p1,p2,p3);
		for(LogicalClock c : clocks) System.out.println(c);
		System.out.println("------");
	}
	
	@Test
	public void test2() {
		ScalarLogicalClock c3 = new ScalarLogicalClock(10);
		ScalarLogicalClock c2 = new ScalarLogicalClock(5);
		c3.setTime(40);
		c2.setTime(25);
		send(c3, c2);
		print(c3, c2);
		
		System.out.println(c3.get().compareTo(c2.get()));
		
//		send(p2, p3);
//		print();
//		
//		send(p3,p2);
//		print();
//		
//		send(p2, p1);
//		print();
	}
}
	
