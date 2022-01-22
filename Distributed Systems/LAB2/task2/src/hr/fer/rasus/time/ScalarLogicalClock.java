package hr.fer.rasus.time;

import hr.fer.rasus.interfaces.LogicalClock;
import hr.fer.rasus.utils.Utils;

public class ScalarLogicalClock implements LogicalClock{
	private TimeLabel time;
	public final int INCREMENT_CONSTANT;
	private String key = "SCALAR";
	
	public ScalarLogicalClock(int period) {
		INCREMENT_CONSTANT = period;
		time = new TimeLabel(key);
	}
	
	public ScalarLogicalClock() {
		this(Utils.getRandomInt(1, 10));
	}
	
	
	@Override
	public int compareTo(TimeLabel other) {
		int otherTime = other.myTime();
		int thisTime  = time.myTime();
		return Integer.compare(thisTime, otherTime);
	}

	@Override
	public TimeLabel get() {
		return time;
	}
	
	public void setTime(int value) {
		TimeLabel t = new TimeLabel(key);
		t.put(key, value);
		this.time = t;
	}

	@Override
	public void increment() {
		int thisTime  = time.myTime();
		thisTime += INCREMENT_CONSTANT;
		time.put(key, thisTime);
	}

	@Override
	public void update(TimeLabel otherTimeLabel) {
		increment();
		int otherTime = otherTimeLabel.myTime();
		int newTime = Math.max(otherTime + 1, time.myTime());
		time.put(key, newTime);
	}

	@Override
	public String toString() {
		return "ScalarLogicalClock [time=" + time + "]";
	}
	

	
}
