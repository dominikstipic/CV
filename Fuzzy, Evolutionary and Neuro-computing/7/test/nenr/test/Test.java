package nenr.test;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;

public class Test {
	
	public static void main(String[] args) throws IOException {
		System.out.println("This goes to the console");
		PrintStream console = System.out;

		File file = new File("./log.txt");
		FileOutputStream fos = new FileOutputStream(file);
		PrintStream ps = new PrintStream(fos);
		System.setOut(ps);
		System.out.println("This goes to out.txt");

		System.setOut(console);
		System.out.println("This also goes to the console");
	}	

}
