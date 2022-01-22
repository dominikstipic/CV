package hr.fer.rasus.interfaces;

import hr.fer.rasus.time.TimeLabel;

public interface LogicalClock extends Comparable<TimeLabel>{
	TimeLabel get();
	void increment();
	void update(TimeLabel otherTimeLabel);
}
