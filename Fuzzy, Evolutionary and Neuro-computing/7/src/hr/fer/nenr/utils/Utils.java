package hr.fer.nenr.utils;

import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.sql.Date;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import au.com.bytecode.opencsv.CSVReader;
import hr.fer.nenr.models.Data;

public class Utils {

	public static String getDate() {
		SimpleDateFormat formatter= new SimpleDateFormat("HH:mm:ss");
		Date date = new Date(System.currentTimeMillis());
		String str = formatter.format(date);
		return str;
	}
	
	public static List<Data> readCSV(String path) {
		List<Data> measurements = new ArrayList<Data>();
		Function<String, Double> parseDouble = s -> s.trim().equals("") ? null : Double.parseDouble(s);
		try (CSVReader csvReader = new CSVReader(new FileReader(path));) {
		    String[] values = null;
		    String line= csvReader.readNext()[0];
		    values = line.split(",");
		    while ((values = csvReader.readNext()) != null) {
		    	List<Double> doubleValues = Arrays.asList(values).stream().
		    			                           map(s -> parseDouble.apply(s)).
		    			                           collect(Collectors.toList());
		    	double x  = doubleValues.get(0);
		    	double y  = doubleValues.get(1);
		    	double o1 = doubleValues.get(2);
		    	double o2 = doubleValues.get(3);
		    	double o3 = doubleValues.get(4);
		    	double example[] = new double[] {x,y};
		    	double oneHot[] = new double[] { o1, o2, o3};
		    	Data data = new Data(example, oneHot);
		    	measurements.add(data);
		    }
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		
		return measurements;
	}
	
	public static List<Integer> randomVector(int size, int sample){
		if(sample > size) throw new IllegalArgumentException("sample > size");
		List<Integer> idx = IntStream.range(0, size).boxed().collect(Collectors.toList());
		Collections.shuffle(idx);
		List<Integer> xs = new ArrayList<>();
		for(int i = 0; i < sample; ++i) {
			int index = idx.get(i);
			xs.add(index);
		}
		return xs;
	} 
	
	public static double randomDouble(double lower, double upper) {
		Random r = new Random(System.nanoTime());
		Double d = r.nextDouble()*upper-lower;
		return d;
	}
	
	public static double[] randomDoubleVector(int n, double lower, double upper) {
		double[] values = new double[n];
		for(int i = 0; i < n; ++i) {
			values[i] = randomDouble(lower, upper);
		}
		return values;
	}
	
	public static int randInt(int lower, int upper) {
		Random r = new Random(System.nanoTime());
		int up = Math.abs(upper+lower);
		Integer d = r.nextInt(up)-lower;
		return d;
	}
	
	public static double[][] toMatrix(double[] arr, int n, int m){
		if(arr.length != n*m) throw new IllegalArgumentException("Sizes doesnt match");
		double[][] matrix = new double[n][m];
		for(int i = 0; i < n; ++i) {
			for(int j = 0; j < m; ++j) {
				int k = i*m + j;
				matrix[i][j] = arr[k];
			}
		}
		return matrix;
	}
	
	public static void printMatrix(double[][] M) {
		int n = M.length;
		int m = M[0].length;
		
		for(int i = 0; i < n; ++i) {
			for(int j = 0; j < m; ++j) {
				String s = M[i][j] + " ";
				System.out.print(s);
			}
			System.out.println();
		}
	}
	
	public static void printMatrix(Double[][] M) {
		int n = M.length;
		int m = M[0].length;
		
		for(int i = 0; i < n; ++i) {
			for(int j = 0; j < m; ++j) {
				String s = M[i][j] + " ";
				System.out.print(s);
			}
			System.out.println();
		}
	}
	
	public static void appendToFile(String line, Path file) {
		try {
		    Files.write(file, line.getBytes(), StandardOpenOption.APPEND);
		}catch (IOException e) {
			e.getStackTrace();
		}
	}
	
	public static List<String> readLines(Path file) {
		List<String> lines = null;
		try {
		    lines = Files.readAllLines(file);
		}catch (IOException e) {
			e.getStackTrace();
		}
		
		return lines;
	}
	
	public static double[] stringToList(String line) {
		line = line.replace("[", "").replace("]", "");
		String[] split = line.split(",");
		double[] vals = new double[split.length];
		for(int i = 0; i < split.length; ++i) {
			vals[i] = Double.parseDouble(split[i]);
		}
		return vals;
	}
	
	
}
