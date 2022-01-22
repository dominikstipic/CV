package hr.fer.rasus.components;
import static hr.fer.rasus.utils.Utils.DATA;
import static hr.fer.rasus.utils.Utils.randomSleep;

import java.io.IOException;
import java.net.SocketException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;

import hr.fer.rasus.dao.RawMeasurement;
import hr.fer.rasus.dao.Tuple2Types;
import hr.fer.rasus.interfaces.Dataset;
import hr.fer.rasus.interfaces.ISubscriber;
import hr.fer.rasus.interfaces.LogicalClock;
import hr.fer.rasus.time.ScalarLogicalClock;
import hr.fer.rasus.time.TimeLabel;
import hr.fer.rasus.time.VectorLogicalClock;
import hr.fer.rasus.utils.LogService;
import hr.fer.rasus.utils.Protocol;
import hr.fer.rasus.utils.SensorUtils;
import hr.fer.rasus.utils.UdpClients;
import hr.fer.rasus.utils.Utils;

public class Sensor implements ISubscriber {
	private String name;
	private int port;
	private Dataset dataset = new MeasurementCollection();
	private UdpServer server;
	private TriggerClock clock; 
	private LogicalClock scalarClock;
	private LogicalClock vectorClock;
	private final int TRIGGER_PERIOD = 5000;
	private boolean active;
	
	public Sensor(String name, int port) {
		this.name = name;
		this.port = port;
	}

	private void initSensor() {
		active=true;
		startUdpServer();
		startTriggerClock();
		
		List<String> allNames = SensorUtils.getAllSensorNames();
		scalarClock = new ScalarLogicalClock();
		vectorClock = new VectorLogicalClock(name, allNames);
	}
	
///////////////////////
	
	@Override
	public void update(String message) {
		if(Protocol.isEnd(message)) {
			// Turn sensor off and his components
			active = false;
			clock.turnOff();
			dataset.clear();
			server.unsubscribe();
			clock.unsubscribe();
		}
		else if(Protocol.isShare(message)) {
			// Other sensors are sharing data
			Tuple2Types<String, RawMeasurement> tuple = Protocol.repackShare(message);
			RawMeasurement m = tuple.t;
			vectorClock.update(m.getVector());
			scalarClock.update(m.getScalar());
			m.setVector(vectorClock.get());
			m.setScalar(scalarClock.get());
			dataset.storeMeasurement(m);
			}
		else if(Protocol.isTrigger(message)) {
			System.out.println("TRIGGER: " + name);
			if(dataset.size() == 0) return;
			List<RawMeasurement> scalarSorted = dataset.scalarSorted();
			List<RawMeasurement> vectorSorted = dataset.vectorSorted();
			RawMeasurement mean = calculateMean();
			
			report(scalarSorted, vectorSorted, mean);
			
			dataset.clear();
		}
		else {
			System.out.println("UNRECOGNIZED");
		}
	}
	
	/**
	 * Report to std-out
	 * @param scalarSorted
	 * @param vectorSorted
	 * @param mean
	 */
	private void report(List<RawMeasurement> scalarSorted, List<RawMeasurement> vectorSorted, RawMeasurement mean) {
		String nl = System.lineSeparator();
		StringBuilder scalarBuilder = new StringBuilder();
		StringBuilder vectorBuilder = new StringBuilder();
		scalarSorted.forEach(r -> {
			Long time = r.getSystemTime();
			scalarBuilder.append(time + ",");
		});
		vectorSorted.forEach(r -> {
			Long time = r.getSystemTime();
			vectorBuilder.append(time + ",");
		});
		
		String s1 = "SCALAR: " + scalarBuilder + nl;
		String s2 = "VECTOR: " + vectorBuilder + nl;
		String s3 = "MEAN: " + mean + nl;
		String s4 = "------" + nl;
		String str = s1+s2+s3+s4;
		System.out.println(str);
	}
	
	
	/**
	 * Calculates mean measurement value for whole period
	 * @return mean measurement value
	 */
	private RawMeasurement calculateMean() {
		Map<String, List<Double>> map = new HashMap<>();
		List<String> names = RawMeasurement.params();
		names.forEach(name -> map.put(name, new LinkedList<>()));
		
		BiConsumer<String, Double> consumer = (s ,d) -> {
			List<Double> list = map.get(s);
			list.add(d);
			map.put(s, list);
		};
		
		for(RawMeasurement raw : dataset) {
			List<Double> values = raw.values();
			for(int i = 0; i < values.size(); ++i) {
				Double d = values.get(i);
				String paramName = names.get(i);
				if(d == null) continue;
				consumer.accept(paramName, d);
			}
		}
		List<Double> means = map.values().stream().map(xs -> Utils.mean(xs)).collect(Collectors.toList());
		RawMeasurement meanMeasure = new RawMeasurement(means);
		return meanMeasure;
	}
	

	/**
	 * Shares data with other sensors
	 */
	private void sendToNeighbours(RawMeasurement measurement) {
		List<Sensor> neighbourSensors = SensorUtils.getNeighbours(this);
		for(Sensor sensor : neighbourSensors) {
			udpClient(measurement, sensor);
		}
	}
	
	/**
	 * Sends measurement toward given sensor
	 * @param data
	 * @param destination
	 */
	private void udpClient(RawMeasurement data, Sensor destination) {
		String json = Utils.toJSON(data);;
		String message = Protocol.shareMessage(this.name, json);		
		int targetPort = destination.port;
		byte[] sendBuf = message.getBytes();
		UdpClients.sendPacket(sendBuf, targetPort);
	}

	/**
	 * Simulates measuring 
	 * @return RawMeasurment
	 */
	private RawMeasurement measure() {
		int activeSeconds = clock.activeSec();
		int N = DATA.size();
		int idx = (int)(activeSeconds % N);
		RawMeasurement raw = DATA.get(idx);
		TimeLabel scalar = scalarClock.get();
		TimeLabel vector = vectorClock.get();
		raw.setVector(vector);
		raw.setScalar(scalar);
		scalarClock.increment();
		vectorClock.increment();
		raw.setSystemTime(System.nanoTime());
		//System.out.println(raw);
		return raw;
	}
	
	/**
	 * Start UDP server in separate thread 
	 */
	private void startUdpServer() {
		try {
			server = new UdpServer(name, port);
		} catch (SocketException e) {
			e.printStackTrace();
		}
		server.subscribe(this);
		Thread serverThread = new Thread(server);
		serverThread.start();
	}
	
	/**
	 * Starts periodic triggering clock. After TRIGGER_PERIOD time 
	 * clock sends TRIGGER message to sensor
	 */
	private void startTriggerClock() {
		clock = new TriggerClock(TRIGGER_PERIOD);
		clock.subscribe(this);
		Thread triggerClock = new Thread(clock);
		triggerClock.start();
	}
	
	
	/**
	 * Starts sensor and his internal components
	 * @throws InterruptedException
	 */
	public void start() throws InterruptedException {
		initSensor();
		while(active) {
			randomSleep(2000, 4000);
			RawMeasurement measurement = measure();
			sendToNeighbours(measurement);
			dataset.storeMeasurement(measurement);
		}
	}
	
	public static void main(String[] args) throws IOException, InterruptedException {
		String name = args[0];
		int port = Integer.parseInt(args[1]);
		
		String helloMsg = String.format("Starting sensor %s on port %d\n", name, port);
		LogService.get().print(helloMsg);
		
		Sensor sensor = new Sensor(name, port);
		sensor.start();
		
		String endMsg = String.format("Sensor %s is disconnecting\n", name);
		LogService.get().print(endMsg);
	}
	
	@Override
	public int hashCode() {
		return Objects.hash(name, port);
	}

	@Override
	public boolean equals(Object obj) {
		if(!obj.getClass().equals(this.getClass())) return false;
		Sensor other = (Sensor) obj;
		return other.name.equals(name) && other.port == port;
	}

	@Override
	public String toString() {
		return "Sensor [name=" + name + ", port=" + port + "]";
	}

	public int getPort() {
		return port;
	}
	
	public String getName() {
		return name;
	}
	
}
