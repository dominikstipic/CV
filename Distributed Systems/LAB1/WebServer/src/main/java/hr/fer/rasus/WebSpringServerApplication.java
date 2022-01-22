package hr.fer.rasus;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;


@SpringBootApplication
public class WebSpringServerApplication{
	public static String LOG_FILE = "./log.txt";
	
	public static void logFileConfig() {
		File file = new File(LOG_FILE);
		System.out.println("CREATING LOG AT: " + file.toString());
		if(!file.exists()) {
			try {
				file.createNewFile();
			} catch (IOException e) {
				System.out.println("Cannot create a log file");
				e.printStackTrace();
			}
		}
		try {
			System.setOut(new PrintStream(file));
		} catch (FileNotFoundException e) {
			System.out.println("Error while configuring std out");
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		//logFileConfig();
		SpringApplication.run(WebSpringServerApplication.class, args);
	}
}
