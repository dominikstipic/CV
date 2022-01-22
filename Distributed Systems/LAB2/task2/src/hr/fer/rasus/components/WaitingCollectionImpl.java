package hr.fer.rasus.components;

import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Semaphore;

import hr.fer.rasus.dao.RawMeasurement;
import hr.fer.rasus.interfaces.WaitingCollection;

public class WaitingCollectionImpl implements WaitingCollection{
	private List<Tuple> list = new LinkedList<>();
	private Semaphore sem = new Semaphore(1);

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
	public void put(RawMeasurement m, String sensorName) {
		Tuple t = new Tuple(m, sensorName);
		lock();
		list.add(t);
		unlock();
	}

	@Override
	public void validate(int measurementHash, String sensorName) {
		for(int i = 0; i < list.size(); ++i) {
			Tuple t = list.get(i);
			if(t.m.hashCode() == measurementHash && sensorName.equals(t.name)) {
				lock();
				list.remove(i);
				unlock();
				break;
			}
		}
		
	}

	@Override
	public List<String> getInvalidated(int hashCode) {
		List<String> names = new LinkedList<>();
		for(int i = 0; i < list.size(); ++i) {
			Tuple t = list.get(i);
			if(t.m.hashCode() == hashCode) {
				names.add(t.name);
			}
		}
		return names;
	}

	@Override
	public boolean isValidated(int hashCode) {
		for(int i = 0; i < list.size(); ++i) {
			Tuple t = list.get(i);
			if(t.m.hashCode() == hashCode) {
				return false;
			}
		}
		return true;
	}

	private static class Tuple {
		public RawMeasurement m;
		public String name;
		
		public Tuple(RawMeasurement m, String name) {
			this.m = m;
			this.name = name;
		}
		
		
	}

	@Override
	public void clear() {
		lock();
		list.clear();
		unlock();
	}

	@Override
	public int size() {
		return list.size();
	}
	
}
