package hr.fer.rasus.subcomponents;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import hr.fer.rasus.components.Sensor;
import hr.fer.rasus.utils.Utils;

public class SensorUtils {
	public final static String SENSOR_DATA = "./data/mjerenja.csv";
	public final static String SENSOR_TABLE = "./data/table.txt";
	private static List<Sensor> ALL_SENSORS;
	
	/**
	 * Reads sensors information from file
	 * @return List of Sensors
	 */
	private static List<Sensor> readSensorTable(){
		List<Sensor> sensors = new ArrayList<>();
		File file = new File(SensorUtils.SENSOR_TABLE);
		try(BufferedReader reader = new BufferedReader(new FileReader(file))){
			List<String> lines = reader.lines().collect(Collectors.toList());
			for(String line : lines) {
				String[] arr = line.split(":");
				String name = arr[0].trim();
				int port = Integer.valueOf(arr[1].trim());
				Sensor sensor = new Sensor(name, port);
				sensors.add(sensor);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return sensors;
	}
	
	/**
	 * Gets all sensors in the distributed system. When first request is made 
	 * sensor information are read from file and then saved. This enables faster access when 
	 * future request are made.
	 * @return List of sensors
	 */
	public static List<Sensor> getAllSensors() {
		if(ALL_SENSORS == null) {
			ALL_SENSORS = readSensorTable();
		}
		return ALL_SENSORS;
	}
	
	/**
	 * Gets the list of sensor names
	 * @return
	 */
	public static List<String> getAllSensorNames() {
		List<Sensor> sensors = getAllSensors();
		List<String> names = sensors.stream().map(sensor -> sensor.getName()).collect(Collectors.toList());
		return names;
	}
	
	
	/**
	 * Gets neighbor sensors information
	 * @return List of neighborhood sensors
	 */
	public static List<Sensor> getNeighbours(Sensor sensor){
		List<Sensor> neighbours = getAllSensors();
		neighbours.remove(sensor);
		return neighbours;
	}
	
	public static Sensor byName(String sensor) {
		List<Sensor> sensors = getAllSensors();
		Sensor s = sensors.stream().filter(sen -> sen.getName().equals(sensor)).findFirst().get();
		return s;
	}
	
	public static List<Sensor> createSensors(int n){
		List<Sensor> sensors = new ArrayList<>();
		for(int i = 0; i < n; ++i) {
			String name = Utils.randomString(7);
			int port = 0;
			try(ServerSocket s = new ServerSocket(0)){
				port = s.getLocalPort();
			} 
			catch (IOException e) {
				e.printStackTrace();
			}
			Sensor sensor = new Sensor(name, port);
			sensors.add(sensor);
		}
		return sensors;
	}

}
