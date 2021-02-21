package hr.fer.nenr.interfaces;

import java.util.Map.Entry;

public interface Dataset<T, S> extends Iterable<Entry<T, S>>{
	void add(T gesture, S label);
	Entry<T, S> get(int idx);
	void remove(T gesture);
	void remove(int idx);
	int getLabelCount(S label);
	void saveData(S label);
}
