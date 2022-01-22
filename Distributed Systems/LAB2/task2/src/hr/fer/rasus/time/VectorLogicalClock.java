package hr.fer.rasus.time;

import static java.lang.Math.max;

import java.util.List;

import hr.fer.rasus.interfaces.LogicalClock;

public class VectorLogicalClock implements LogicalClock{
	private TimeLabel time;
	public final int INCREMENT_CONSTANT = 1;
	
	public VectorLogicalClock(String thisName, List<String> otherNames) {
		time = new TimeLabel(thisName, otherNames);
	}
	
	public VectorLogicalClock(TimeLabel label) {
		this.time = label;
	}

	
	@Override
	public int compareTo(TimeLabel other) {
		TimeLabel thisLabel = get();
		return thisLabel.compareTo(other);
	}

	@Override
	public TimeLabel get() {
		return time;
	}
	
	public void set(TimeLabel time) {
		this.time = time;
	}

	@Override
	public void increment() {
		int value = time.myTime();
		String me = time.getSource();
		time.put(me, value+1);
	}

	@Override
	public void update(TimeLabel otherTimeLabel) {
		increment();
		for(String key : time.getMap().keySet()) {
			int thisTime  = time.forSensor(key);
			int otherTime = otherTimeLabel.forSensor(key);
			int newTime   = max(thisTime, otherTime);
			time.put(key, newTime);
		}
	}

	@Override
	public String toString() {
		return "VectorLogicalClock [time=" + time + "]";
	}

	
	
}
