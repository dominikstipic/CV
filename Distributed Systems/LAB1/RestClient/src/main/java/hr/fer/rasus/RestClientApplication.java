package hr.fer.rasus;

import static hr.fer.rasus.utils.Utils.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import org.springframework.web.client.RestTemplate;
import hr.fer.rasus.client.Sensor;
public class RestClientApplication{

	
	public static void initServer() {
		String url1 = "http://localhost:8080/measurment";
		String url2 = "http://localhost:8080/register";
		new RestTemplate().delete(url1);
		new RestTemplate().delete(url2);
	}
	
	public static void main(String[] args) throws InterruptedException {
		initServer();
		int numberOfSensors = 3;
		
		//SENSORS
		List<Sensor> sensors = new ArrayList<>();
		for(int i = 0; i < numberOfSensors; ++i) {
			Sensor s = new Sensor();
			sensors.add(s);
		}
		
		//CRETING SENSOR THREADS
		List<Thread> threads = new ArrayList<>();
		sensors.forEach(s -> {
			Thread t = new Thread(s);
			t.setName(s.getUsername());
			threads.add(t);
		});
		
		//RUNNING ALL THREADS
		threads.forEach(t -> t.start());
		
		//USER THREAD
		Thread.sleep(1000);
		try(Scanner s = new Scanner(System.in)){
			while(anyActive(sensors)) {
				Integer index = input(s);
				if(index >= 0 && index < sensors.size()) {
					sensors.get(index).deactivate();
				}
				else if(index == -1) {
					sensors.forEach(sen -> sen.deactivate());
				}
				Thread.sleep(500);
				}
			}
		threads.forEach(t -> {
			try {
				t.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		});
		System.out.println("SYSTEM CLOSING!");
		}
		
	}

		
	

