package hr.fer.rasus.utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class LogService{
	private static String PATH = "./log.txt";
	private static LogService log;
	private PrintWriter writer;
	
	private LogService(){}
	
	public static LogService get(){
		if(log == null) {
			log = new LogService();
		}
		return log;
	}
	
	public static String getPath() {
		return PATH;
	}
	
	public static void setPath(String path) {
		PATH = path;
	}
	
	public void print(String line) {
		try {
			open(true);
			writer.print(line);
			writer.print(System.lineSeparator());
			writer.print("-----");
			writer.print(System.lineSeparator());
		}
		catch (Exception e) {
			e.printStackTrace();
		} 
		finally {
			close();
		}
	}
	
	public void delete() {
		try {
			open(false);
			close();
		} finally {
			close();
		}
	}
	
	public void nl() {
		print("\n");
	}

	/**
	 * Closes resources which LogService uses
	 */
	private void close(){
		writer.close();
	}

	/**
	 * Opens underline PrintWriter which enables file logging
	 */
	private void open(boolean append) {
		File file = new File(PATH);
		try {
			writer = new PrintWriter(new BufferedWriter(new FileWriter(file, append)), true);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
