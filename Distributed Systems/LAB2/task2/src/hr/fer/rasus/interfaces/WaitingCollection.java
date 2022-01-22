package hr.fer.rasus.interfaces;

import java.util.List;

import hr.fer.rasus.dao.RawMeasurement;

public interface WaitingCollection {
	void put(RawMeasurement m, String sensorName);
	void validate(int measurementHash, String sensorName);
	List<String> getInvalidated(int hashCode);
	boolean isValidated(int hashCode);
	void clear();
	int size();
}
