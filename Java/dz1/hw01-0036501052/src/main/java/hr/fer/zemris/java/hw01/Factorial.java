package hr.fer.zemris.java.hw01;

import java.util.Scanner;


/**
 * Class which implements mathematical operation -> Factorial( n! )
 * @author Dominik Stipić
 * @version 1.0
 */
public class Factorial {
	
	public static final int MAX_INPUT = 20; 
	public static final int MIN_INPUT = 0;
	
	/**
	 * Calculates factorial of input which is non-negative integer,according to formula:
	 * n! = n * (n-1) * ... * 1 where 0! = 1
	 * @param n Non-negative number where : n >= 0 and n < 20  
	 * @return Factorial of n 
	 * @throws IllegalArgumentException if the argumet isn't in defined interval
	 */
	public static int factorial(int n) {
		if(isInInterval(n) == false) {
			throw new IllegalArgumentException("Argument isn't in defined interval");
		}
		if (n == 0) {
			return 1;
		}
		return n * factorial(n - 1);
	}
 
	
	/**
	 * Method checks if the number is in defined interval
	 * @param num Number which we check
	 * @return true - number is in interval
	 * 		   false - number isn't in interval	  
	 */
	public static boolean isInInterval(int num) {
		return (num >= MIN_INPUT && num < MAX_INPUT) ? true : false;
	}
	
	/**
	 * Method which is automatically called when a program starts.
	 * @param args Arguments from command-line interface
	 */
	public static void main(String[] args) {
		try (Scanner s = new Scanner(System.in)) {
			
			while (true) {
				System.out.println("Enter your input >");
				String str = s.next().toLowerCase();
				if (str.equals("end") || str.equals("kraj")) {	//sign for stopping interaction user-program
					System.out.println("Goodbye");
					break;
				}

				try {
					int number = Integer.parseInt(str);
					System.out.println(number + "! = " + factorial(number));

				} catch (NumberFormatException exc) {
					System.out.println(str + " isn't integer number");
				} catch (IllegalArgumentException exc) {
					System.out.println(exc.getMessage());
				}
			}
		}
	}
}
