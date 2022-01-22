package hr.fer.rasus.utils;

import java.io.FileReader;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.commons.lang3.RandomStringUtils;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import au.com.bytecode.opencsv.CSVReader;
import hr.fer.rasus.dao.RawMeasurement;


public class Utils {
	public static List<RawMeasurement> DATA = readCsv(SensorUtils.SENSOR_DATA);

	/**
	 * Random sleep for time in a given interval. Intervals are expressed as milliseconds
	 * @param lower
	 * @param upper
	 */
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
	
	public static int getRandomInt(int lower, int upper) {
		long time = System.nanoTime();
		Random r = new Random(time);
		int rand = r.nextInt(upper)+lower;
		return rand;
	}
	
	public static List<RawMeasurement> readCsv(String path) {
		List<RawMeasurement> measurements = new ArrayList<RawMeasurement>();
		Function<String, Double> parseDouble = s -> s.trim().equals("") ? null : Double.parseDouble(s);
		
		try (CSVReader csvReader = new CSVReader(new FileReader(path));) {
		    String[] values = null;
		    values = csvReader.readNext();
		    while ((values = csvReader.readNext()) != null) {
		    	List<Double> doubleValues = Arrays.asList(values).stream().
		    			                           map(s -> parseDouble.apply(s)).
		    			                           collect(Collectors.toList());
		    	RawMeasurement m = new RawMeasurement(doubleValues);
		    	measurements.add(m);
		    }
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		
		return measurements;
	}
	
	public static String randomString(int lenght) {
		String s = RandomStringUtils.random(lenght, true, true);
		return s;
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
	
	public static <T> String toJSON(T obj) {
		ObjectMapper mapper = new ObjectMapper();
		String json = null;
		try {
			json = mapper.writeValueAsString(obj);
		} catch (JsonProcessingException e) {
			e.printStackTrace();
		}
		return json;
	}
	
	
	public static <T> T fromJSON(String json, Class<T> objClass) {
		ObjectMapper mapper = new ObjectMapper();
		T out = null;
		try {
			out = mapper.readValue(json, objClass);
		} catch(Exception e) {
			e.printStackTrace();
			System.out.println(json);
			System.out.println("CLASS " + objClass);
		}
		return out;
	}
	
	public static String currentDate() {
		SimpleDateFormat formatter= new SimpleDateFormat("yyyy-MM-dd 'at' HH:mm:ss z");
		Date date = new Date(System.currentTimeMillis());
		String str = formatter.format(date);
		return str;
	}
	
	public static double mean(List<Double> list){
		int N = list.size();
		double sum = list.stream().mapToDouble(d -> (double) d).sum();
		return sum/N;
	}
	
	public static <S,T> Map<S, T> zip(List<S> keys, List<T> vals){
		Map<S, T> map = new HashMap<>();
		for(int i = 0; i < keys.size(); ++i) {
			S key = keys.get(i);
			T val    = vals.get(i);
			map.put(key, val);
		}
		return map;
	} 
}
