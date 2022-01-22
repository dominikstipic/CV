package hr.fer.rasus.utils;

import java.io.FileReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.stream.Collectors;

import org.apache.commons.lang3.RandomStringUtils;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.opencsv.bean.CsvToBeanBuilder;

import hr.fer.rasus.client.ClientServer;
import hr.fer.rasus.client.Sensor;

public class Utils {
	
	public static boolean anyActive(List<Sensor> sensors) {
		for(Sensor s : sensors) {
			if(s.isActive()) return true;
		}
		return false;
	}
	
	public static String activeString(List<Sensor> list) {
		String symbol = list.stream().
				 		     map(s -> s.isActive()).
						     map(b -> b == true? "O" : "X" ).
						     collect(Collectors.joining(",", "[", "]"));
		return symbol;
	}
	
	public static void randomSleep(int lower, int upper) {
		try {
			int delta = upper - lower;
			int randTime_milli = new Random().nextInt(lower) + delta;
			Thread.sleep(randTime_milli);
		} 
		catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	public static double getRandomDouble(double lower, double upper) {
		Random r = new Random();
		double randomValue = lower + (upper - lower)*r.nextDouble();
		return randomValue;
	}
	
	
	public static String getRandomString(int lenght) {
		String s = RandomStringUtils.random(lenght, true, true);
		return s;
	}
	
	
	public static Integer input(Scanner s) {
		Integer result=0;
		while(true) {
			System.out.print("> ");
			try {
				result = Integer.parseInt(s.nextLine());
				break;
			} catch (Exception e) {
				System.out.println("Wrong input");
			}
		}
		return result;
	}
	
	public static String readLine(Scanner s) {
		System.out.print("> ");
		String line="";
		try {
			line = s.nextLine();
		} catch (Exception e) {
			System.out.println("Wrong input");
		}
		return line;
	}

	public static <T> List<T> flattenList(List<List<T>> list){
		List<T> result = new ArrayList<>();
		for(List<T> xs : list) {
			for(T x : xs) {
				result.add(x);
			}
		}
		return result;
	}
	
	public static <V> List<V> readCsv(String path, Class<V> objClass) {
		List<V> readings = null;
		try {
			readings = new CsvToBeanBuilder<V>(new FileReader(path))
					.withType(objClass).build().parse();
		} catch (Exception e) {
			System.out.println("Error while reading measurement dataset");
		}
		return readings;
	}
	
	public static String randomIP() {
		Random r = new Random();
		String ip = r.nextInt(256) + "." + r.nextInt(256) + "." + r.nextInt(256) + "." + r.nextInt(256); 
		return ip;
	}
	
	public static Float avarageNumbers(List<? extends Number> list){
		List<? extends Number> noNullList = list.stream().filter(n -> n!= null).collect(Collectors.toList());
		if(noNullList.size() == 0) {
			return null;
		}
		Double avgDouble = noNullList.stream().mapToDouble(n -> n.doubleValue()).average().getAsDouble();
		Float avgFloat = avgDouble.floatValue();
		return avgFloat;
	}
	
	public static <T> T fromJson(String json, Class<T> objClass) {
		ObjectMapper mapper = new ObjectMapper();
		T obj = null;
		try {
			obj = mapper.readValue(json, objClass);
		} catch (JsonProcessingException e) {
			System.out.println("JSON TO OBJECT ERROR!");
			e.printStackTrace();
		}
		return obj;
	}
	
	public static <T> String toJson(T obj) {
		ObjectMapper mapper = new ObjectMapper();
		String json = "";
		try {
			json = mapper.writeValueAsString(obj);
		} catch (JsonProcessingException e) {
			System.out.println("OBJECT TO JSON ERROR!");
			e.printStackTrace();
		}
		return json;
	}
	
	public static void socketSend(String line, int port) {
		try (Socket clientSocket = new Socket(ClientServer.HOST_NAME, port);
			 PrintWriter outToServer = new PrintWriter(
		                               new OutputStreamWriter(clientSocket.getOutputStream()), true);) {
				outToServer.println(line);
			}
		catch (Exception e) {
			System.out.println("ERROR WHILE CONNECTING");
		}
	}
}
