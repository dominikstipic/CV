package hr.fer.rasus.client;

import hr.fer.rasus.client.exceptions.RegistrationError;
import hr.fer.rasus.dao.Measurement;
import hr.fer.rasus.dao.SensorDescription;
import hr.fer.rasus.dao.UserAddress;

public interface RestInterface {
	void registerSensor(SensorDescription description) throws RegistrationError;
	UserAddress searchNeighbour();
	void storeMeasurement(Measurement measurement);
}
