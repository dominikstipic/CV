package hr.fer.rasus.main;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;

import hr.fer.rasus.components.Sensor;
import hr.fer.rasus.subcomponents.SensorUtils;
import hr.fer.rasus.utils.LogService;
import hr.fer.rasus.utils.UdpClients;
import hr.fer.rasus.utils.Utils;

public class Main {
	public static List<Sensor> sensors;

	public static void execute(String ...args) {
		String javaHome = System.getProperty("java.home");
        String javaBin = javaHome + 
        		File.separator + "bin" +
                File.separator + "java";
        String classpath = System.getProperty("java.class.path");
        String className = Sensor.class.getName();
        
        List<String> command = List.of(javaBin, "-cp", classpath, className, args[0], args[1]);
        ProcessBuilder builder = new ProcessBuilder(command);
        try {
			builder.inheritIO().start();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void client() throws UnknownHostException, SocketException {
		List<Integer> ports = sensors.stream().map(s -> s.getPort()).collect(Collectors.toList());
		try(Scanner s = new Scanner(System.in)){
			while(true) {
				System.out.println(">");
				String line = s.nextLine();
				for(Integer port : ports) {
					byte[] sendBuf = line.getBytes();
					UdpClients.sendPacket(sendBuf, port, false);
				}
				if(line.equals("END")) break;
			}
		}
	}
	
	public static void createSensorTable() throws IOException {
		try(PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(SensorUtils.SENSOR_TABLE, false)), true);){
			for(Sensor s : sensors) {
				String line = String.format("%s : %d\n", s.getName(), s.getPort());
				writer.print(line);
			}
		}
	}
	
	public static void main(String[] args) throws IOException, InterruptedException {
		final int SENSOR_NUM = 2;
		sensors = SensorUtils.createSensors(SENSOR_NUM);
		LogService.get().delete();
		LogService.get().print(Utils.currentDate());
		LogService.get().nl();
		createSensorTable();
		for(int i = 0; i < sensors.size(); ++i) {
			String arg1 = sensors.get(i).getName();
			String arg2 = String.valueOf(sensors.get(i).getPort());
			execute(arg1, arg2);
		}
		client();
	}

}
