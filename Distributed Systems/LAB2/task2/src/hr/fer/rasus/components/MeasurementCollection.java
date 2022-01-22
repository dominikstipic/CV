package hr.fer.rasus.components;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Semaphore;

import hr.fer.rasus.dao.RawMeasurement;
import hr.fer.rasus.interfaces.Dataset;

public class MeasurementCollection implements Dataset{
	private List<RawMeasurement> data = new ArrayList<>();
	private Semaphore sem = new Semaphore(1);
	
	@Override
	public Iterator<RawMeasurement> iterator() {
		List<RawMeasurement> xs = new ArrayList<>(data);
		return xs.iterator();
	}

	private void lock() {
		try {
			sem.acquire();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	private void unlock() {
		sem.release();
	}
	
	
	@Override
	public void storeMeasurement(RawMeasurement measurement) {
		lock();
		data.add(measurement);
		unlock();
	}

	@Override
	public void clear() {
		lock();
		data.clear();
		unlock();
	}

	@Override
	public List<RawMeasurement> scalarSorted() {
		Comparator<RawMeasurement> compar = (r1,r2) -> r1.getScalar().compareTo(r2.getScalar());
		return sort(compar);
	}
	
	@Override
	public List<RawMeasurement> vectorSorted() {
		Comparator<RawMeasurement> compar = (r1,r2) -> r1.getVector().compareTo(r2.getVector());
		return sort(compar);
	}
	
	private List<RawMeasurement> sort(Comparator<RawMeasurement> compar){
		List<RawMeasurement> xs = new ArrayList<>(data);
		Collections.sort(xs, compar);
		return xs;
	}

	@Override
	public boolean contains(RawMeasurement m) {
		return data.contains(m);
	}

	@Override
	public int size() {
		return data.size();
	}

	
	

	
	
	
}
