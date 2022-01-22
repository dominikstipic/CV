package hr.fer.rasus.interfaces;

import java.util.List;

import hr.fer.rasus.dao.RawMeasurement;

public interface Dataset extends  Iterable<RawMeasurement>{
	void storeMeasurement(RawMeasurement measurement);
	void clear();
	List<RawMeasurement> scalarSorted();
	List<RawMeasurement> vectorSorted();
	boolean contains(RawMeasurement m);
	int size();
}
