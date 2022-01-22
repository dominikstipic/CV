package hr.rasus.tests;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.junit.Test;

import hr.fer.rasus.dao.RawMeasurement;
import hr.fer.rasus.interfaces.LogicalClock;
import hr.fer.rasus.subcomponents.SensorUtils;
import hr.fer.rasus.time.ScalarLogicalClock;
import hr.fer.rasus.time.TimeLabel;
import hr.fer.rasus.time.VectorLogicalClock;
import hr.fer.rasus.utils.Utils;


public class MeasurementTest {
	List<String> allNames = SensorUtils.getAllSensorNames();
	private LogicalClock scalarClock;
	private LogicalClock vectorClock;
	{
		scalarClock = new ScalarLogicalClock();
		vectorClock = new VectorLogicalClock(allNames.get(0), allNames);
	}
	public RawMeasurement get(int i) {
		RawMeasurement m = Utils.DATA.get(i);
		TimeLabel l1 = scalarClock.get();
		TimeLabel l2 = vectorClock.get();
		m.updateTime(l1, l2);
		return m;
	}
	
	public void inc() {
		scalarClock.increment();
		vectorClock.increment();
	}
	
	public List<RawMeasurement> getList(int n) {
		List<RawMeasurement> xs = new ArrayList<>();
		for(int i = 0; i < n; ++i) {
			RawMeasurement m = Utils.DATA.get(i);
			TimeLabel l1 = scalarClock.get();
			TimeLabel l2 = vectorClock.get();
			m.updateTime(l1, l2);
			xs.add(m);
			inc();
		}
		return xs;
	}
	
	public void test1() {
		RawMeasurement m = get(0);
		System.out.println(m);
		System.out.println("-----");
		String json = Utils.toJSON(m);
		System.out.println(json);
		System.out.println("------");
		RawMeasurement m1 = Utils.fromJSON(json, RawMeasurement.class);
		System.out.println(m1);
	}
	
	@Test
	public void test2() {
		List<RawMeasurement> xs = getList(10);
		System.out.println(xs);
		System.out.println("---");
		Comparator<RawMeasurement> compar = (r1,r2) -> r1.getScalar().compareTo(r2.getScalar());
		Collections.sort(xs, compar);
	}
}
