package hr.fer.rasus.client;
import static hr.fer.rasus.utils.Utils.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.Socket;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.springframework.http.converter.json.MappingJackson2HttpMessageConverter;
import org.springframework.util.SocketUtils;
import org.springframework.web.client.RestTemplate;
import hr.fer.rasus.client.exceptions.RegistrationError;
import hr.fer.rasus.dao.Measurement;
import hr.fer.rasus.dao.RawMeasurement;
import hr.fer.rasus.dao.SensorDescription;
import hr.fer.rasus.dao.UserAddress;
import hr.fer.rasus.utils.Utils;

public class Sensor implements RestInterface, Runnable{
	public static String DATA_PATH = "lab/mjerenja.csv";
	private static double LONGITUDE_MIN = 15.87;
	private static double LONGITUDE_MAX = 16.00;
	private static double LATITUDE_MIN  = 45.75;
	private static double LATITUDE_MAX  = 45.85;
	private static int USERNAME_LENGTH  = 7;
	private String baseUrl = "http://localhost:8080";
	
	private RestTemplate restTemplate;
	private List<RawMeasurement> data;
	
	private double longitude, latitude;
	private String username, ipAdress;
	private int port;
	private Instant creationTime;
	private boolean isActive=true;
	
	private ClientServer job;
	private Thread serverThread;
	
	public Sensor() {
		restTemplate = new RestTemplate();
		restTemplate.getMessageConverters().add(new MappingJackson2HttpMessageConverter());
		this.data = readCsv(DATA_PATH, RawMeasurement.class); 
		
		this.longitude = getRandomDouble(LONGITUDE_MIN, LONGITUDE_MAX);
		this.latitude = getRandomDouble(LATITUDE_MIN, LATITUDE_MAX);
		this.username = getRandomString(USERNAME_LENGTH);
		this.ipAdress = randomIP();
		this.port = SocketUtils.findAvailableTcpPort();
		creationTime = Instant.now();
		
		SensorDescription description = this.getSensorDescription();
		try {
			registerSensor(description);
		} catch (RegistrationError e) {
			System.out.println(e);
		}
		
		job = new ClientServer(port);
		
	}
	

	public SensorDescription getSensorDescription() {
		SensorDescription sd = new SensorDescription(username, ipAdress, port, latitude, longitude);
		return sd;
	} 
	
	public String getUsername() {
		return username;
	}
	
	public boolean isActive() {
		return isActive;
	}

	/**
	 * Measures the environment parameters
	 * @return encapsulated parameter names and their value
	 */
	public RawMeasurement measure() {
		Instant currentTime = Instant.now();
		long ns = Duration.between(creationTime, currentTime).toNanos();
		long sec = (long) ((long) ns / 1.0e9);
		int rb = (int)(sec % 100) + 2;
		return this.data.get(rb);
	}
	
	@Override
	public void registerSensor(SensorDescription description) throws RegistrationError {
		try {
			restTemplate.postForEntity(baseUrl + "/register", description, String.class);
		} catch (Exception e) {
			throw new RegistrationError("Error while registrating sensor");
		}
	}
	
	@Override
	public UserAddress searchNeighbour() {
		String url = baseUrl + "/search?username=" + username;
		UserAddress closestAddress = restTemplate.getForObject(url, UserAddress.class);
		return closestAddress;
	}
	
	@Override
	public void storeMeasurement(Measurement measurement) {
		restTemplate.postForEntity(baseUrl + "/measurment", measurement, String.class);
	}

	/**
	 * Deactivates the sensor measurement thread and a server thread. 
	 */
	public void deactivate() {
		isActive = false;
		job.setRunningFlag(false);
		
		//System.out.println("MY " + this.port + ", CLOSING " + p);
		
		socketSend("CLOSE", port);
		try {
			serverThread.join();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	private List<Measurement> fromRaw(RawMeasurement raw){
		List<Double> values = raw.values();
		List<String> params   = raw.params();
		List<Measurement> measurements = new ArrayList<>();
		for(int i = 0; i < values.size(); ++i) {
			Double val = values.get(i);
			Float floatValue = val == null ? null : val.floatValue();
			Measurement m = new Measurement(this.username, params.get(i), floatValue);
			measurements.add(m);
		}
		return measurements;
	}

	//////////////////////////////////////////////
	
	/**
	 * Establishing TCP connection toward TCP server 
	 * @param n number of messages that will be exchanged between client and server
	 * @param port port number of server
	 * @return list of retrieved data from server
	 * @throws IOException
	 */
	public List<String> communication(int n, int port) throws IOException {
		List<String> strings = new ArrayList<>();
		
		try (Socket clientSocket = new Socket(ClientServer.HOST_NAME, port);
			 PrintWriter outToServer = new PrintWriter(
		                               new OutputStreamWriter(clientSocket.getOutputStream()), true);
			 BufferedReader inFromServer = new BufferedReader(
	                                      new InputStreamReader(clientSocket.getInputStream()));) {
			System.out.println("CLIENT " + clientSocket);
			for(int i = 0; i < n; ++i) {
				outToServer.println("MEASURE");
				String s = inFromServer.readLine();
				System.out.println(s);
				strings.add(s);
			}
		}
		catch (Exception e) {
			System.out.println("ERROR WHILE CONNECTING");
		}
		return strings;
	}
	
	/**
	 * Retrieving measurements from closest neighbor. 
	 * The measures are retrieved by opening TCP connection to neighbor's socket
	 * @return List of closest neighbors' measures
	 */
	private List<Measurement> measurementsFromNeighbor(){
		UserAddress adr = searchNeighbour();
		int port = adr.getPort();
		List<RawMeasurement> measurements = new ArrayList<>();
		
		int numberOfMeasures = new Random().nextInt(7)+1;
		try {
			System.out.println("FROM: " + this.port + ", TO: " + port);
			List<String> strings = communication(numberOfMeasures, port);
			System.out.println(strings);
			measurements = strings.stream().
	                               map(s -> fromJson(s, RawMeasurement.class)).
	                               collect(Collectors.toList());
		} catch (IOException e) {
			e.printStackTrace();
			System.out.println("ERROR WHILE SETTING TCP CONNECTION ON PORT " + port);
		} 
		List<Measurement> result = new ArrayList<>();
		for(RawMeasurement r : measurements) {
			List<Measurement> list = fromRaw(r);
			result.addAll(list);
		}
		System.out.println(result);
		return result;
	}
	
	/**
	 * Averages all measurements in order to reduce the noise. 
	 * @param measures neighbor and this sensor measures
	 * @return averaged measured for each parameter
	 */
	private List<Measurement> averageMeasures(List<Measurement> measures) {
		Map<String, List<Float>> map = new LinkedHashMap<>();

		for(Measurement m : measures) {
			String key = m.getUsername();
			Float val  = m.getAvarageValue();
			if(map.containsKey(key)) {
				List<Float> vals = map.get(key);
				vals.add(val);
				map.put(key, vals);
			}
			else {
				List<Float> list = new ArrayList<>();
				list.add(val);
				map.put(key, list);
			}
		}
		List<Measurement> avaraged = new ArrayList<>();
		for(Entry<String, List<Float>> e : map.entrySet()) {
			String param = e.getKey();
			List<Float> floats = e.getValue();
			Float avgValue = avarageNumbers(floats);
			Measurement m = new Measurement(username, param, avgValue);
			avaraged.add(m);
		}
		return avaraged;
	} 
	
	@Override
	public void run() {
		String threadName = Thread.currentThread().getName();
		System.out.println("RUNNING: " + threadName);
		serverThread = new Thread(job);
		serverThread.setName("SERVER_"+threadName);
		serverThread.start();
		
		while(isActive) {
			Utils.randomSleep(4000,6000);
			RawMeasurement raw = measure();
			List<Measurement> myMeasures = fromRaw(raw);
			
//			RawMeasurement r = measure();
//			List<Measurement> neighbourMeasures = fromRaw(r);
			
			List<Measurement> neighbourMeasures = measurementsFromNeighbor();
			List<Measurement> allMeasures = Stream.concat(myMeasures.stream(), neighbourMeasures.stream()).collect(Collectors.toList());
			List<Measurement> averaged = averageMeasures(allMeasures);
			averaged.forEach(m -> storeMeasurement(m));
		}
	}

	
	
	


	
}
