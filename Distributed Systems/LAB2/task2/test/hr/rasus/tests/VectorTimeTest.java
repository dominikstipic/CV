package hr.rasus.tests;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import hr.fer.rasus.interfaces.LogicalClock;
import hr.fer.rasus.time.TimeLabel;
import hr.fer.rasus.time.VectorLogicalClock;
import hr.fer.rasus.utils.Utils;

public class VectorTimeTest {
	private VectorLogicalClock p1;
	private VectorLogicalClock p2;
	private VectorLogicalClock p3;
	private List<String> names = List.of("p1", "p2", "p3");
	
	public VectorTimeTest() {
		initClocks();
	}
	
	private void initClocks() {
		p1 = new VectorLogicalClock("p1", names);
		p2 = new VectorLogicalClock("p2", names);
		p3 = new VectorLogicalClock("p3", names);
	}

//	private LogicalClock initClock(int id, int[] xs) {
//		TimeLabel p2 = new TimeLabel(xs);
//		p2.setId(id);
//		return new VectorLogicalClock(p2);
//	}
	
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
	
	public void textBookTest() {
		send(p1, p2);
		print(p1, p2);
		
		send(p3, p2);
		print();
		
		send(p2,p1);
		print();
		
		send(p2, p3);
		print();
		
		send(p1,p2);
		print();
		
		send(p3,p1);
		print();
		
		send(p2,p3);
		print();
		
		send(p3,p1);
		print();
		
	}
	
	private VectorLogicalClock build(String name, List<Integer> list) {
		Map<String, Integer> m1 = Utils.zip(names, list);
		TimeLabel l1 = new TimeLabel(name, m1);
		VectorLogicalClock c = new VectorLogicalClock(name, names);
		c.set(l1);
		return c;
	}
	
	@Test
	public void test1() {
		VectorLogicalClock c2 = build("p2", Arrays.asList(3,6,1));
		VectorLogicalClock c3 = build("p3", Arrays.asList(1,4,3));
		print(c2, c3);
		System.out.println(c2.get().compareTo(c3.get()));
		System.out.println(c3.get().compareTo(c2.get()));
		
	    c2 = build("p2", Arrays.asList(3,3,1));
	    c3 = build("p3", Arrays.asList(3,5,1));
		print(c2, c3);
		System.out.println(c2.get().compareTo(c3.get()));
		System.out.println(c3.get().compareTo(c2.get()));
	}
//	
//	public void test2() {
//		int procId = 0;
//		LogicalClock clock = initClock(procId, new int[] {3,3,1});
//		TimeLabel p1 = new TimeLabel(1,4,3);
//		clock.update(p1);
//		
//		TimeLabel target = TimeLabel.of(4,4,3);
//		target.setId(0);
//		
//		TimeLabel actual = clock.get();
//		assertEquals(target, actual);
//	}
//	
//	public void test3() {
//		int procId = 0;
//		TimeLabel a = new TimeLabel(2,3,1);
//		TimeLabel b = new TimeLabel(3,3,1);
//		a.setId(procId);
//		b.setId(procId);
//		
//		int x = a.compareTo(b);
//		int y = b.compareTo(a);
//		System.out.println(x + " " + y);
//	}
//	
//	public void test4() {
//		TimeLabel a = new TimeLabel(2,3,1);
//		TimeLabel b = new TimeLabel(3,6,1);
//		a.setId(0);
//		b.setId(1);
//		
//		int x = a.compareTo(b);
//		int y = b.compareTo(a);
//		System.out.println(x + " " + y);
//	}
//	
//	public void test5() {
//		int procId = 0;
//		TimeLabel a = new TimeLabel(3,6,1);
//		TimeLabel b = new TimeLabel(1,4,3);
//		a.setId(procId);
//		b.setId(procId);
//		
//		int x = a.compareTo(b);
//		int y = b.compareTo(a);
//		System.out.println(x + " " + y);
//		assertEquals(x, 0);
//		assertEquals(y, 0);
//	}
//	
//	public void test6() {
//		int procId = 1;
//		LogicalClock clock = initClock(procId, new int[] {1,4,1});
//		clock.increment();
//		
//		TimeLabel target = TimeLabel.of(1,5,1);
//		target.setId(procId);
//		
//		System.out.println(clock.get());
//		assertEquals(clock.get(), target);
//	}
}
