package hr.fer.rasus.contoller;

import static java.lang.Math.atan2;
import static java.lang.Math.cos;
import static java.lang.Math.pow;
import static java.lang.Math.sin;
import static java.lang.Math.sqrt;

import java.time.Instant;
import java.util.List;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import hr.fer.rasus.dao.Measurement;
import hr.fer.rasus.dao.SensorDescription;
import hr.fer.rasus.dao.UserAddress;
import hr.fer.rasus.repo.MeasurementRepository;
import hr.fer.rasus.repo.SensorRepository;


@RestController
public class MeasurementController {
	@Autowired
	private MeasurementRepository dataRepository;
	@Autowired
	private SensorRepository sensorRepository;
	
	private double sensorDistance(SensorDescription s1, SensorDescription s2) {
		double long1 = s1.getLongitude();
		double long2 = s2.getLongitude();
		double lat1  = s1.getLatitude();
		double lat2  = s2.getLatitude();
		
		int R = 6371;
		double dlon = long2 - long1;
		double dlat = lat1 - lat2;
		double a = pow((sin(dlat/2)),2) + cos(lat1)*cos(lat2)*pow(sin(dlon/2),2);
		double c = 2*atan2(sqrt(a), sqrt(1-a));
		double d = R*c;
		return d;
	}
	
	@GetMapping("/")
	public String mainPage() {
		Instant inst = Instant.now();	
		String  s = "Current time: " + inst;
		return s;
	}
	
	//tested
	@PostMapping("/register")
	public void registerSensor(@RequestBody SensorDescription description) { boolean bool = sensorRepository.existsById(description.getUsername());
		if(!bool) {
			
			sensorRepository.save(description);
		}
	}
	
	//tested
	@GetMapping("/register")
	public List<SensorDescription> getSensors() {
		List<SensorDescription> sensors =  sensorRepository.findAll();
		return sensors;
	}
	
	@DeleteMapping("/register")
	public void deleteDescription() {
		this.sensorRepository.deleteAll();
	}
	
	//tested
	@GetMapping("/search")
	public UserAddress searchNeighbour(@RequestParam String username) {
		boolean bool = sensorRepository.existsById(username);
		if(!bool) throw new IllegalArgumentException("The sensor with given username doesn't exist");
		
		List<SensorDescription> otherSensors = sensorRepository.findAll();
		SensorDescription targetSensor = sensorRepository.getOne(username);
		otherSensors.remove(targetSensor);
		
		List<Double> distances = otherSensors.stream().map(s -> sensorDistance(targetSensor, s)).collect(Collectors.toList());
		Double min = distances.stream().min((d1,d2) -> Double.compare(d1, d2)).get();
		int index = distances.indexOf(min);
		SensorDescription closestSensor = otherSensors.get(index);
		UserAddress adr = new UserAddress(closestSensor.getUsername(), closestSensor.getIpAddress(), closestSensor.getPort());
		System.out.format("SEARCHING : (TARGET=%s, CLOSEST=%s, DISTANCE=%s)",username, closestSensor.getUsername(), min );
		return adr;
	}
	
	//tested
	@GetMapping("/measurment")
	public List<Measurement> getMeasurements() {
		List<Measurement> data = dataRepository.findAll();
		return data;
	}
	
	//tested
	@PostMapping("/measurment")
	public void storeMeasurment(@RequestBody Measurement measurement) {
		System.out.println("MEASUREMENT: " + measurement);
		this.dataRepository.save(measurement);
	}
	
	@DeleteMapping("/measurment")
	public void deleteMeasurments() {
		this.dataRepository.deleteAll();
	}
}
