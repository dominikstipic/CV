package hr.fer.zemris.java.hw01;

import java.util.InputMismatchException;
import java.util.Scanner;


/**
 * Class which implements methods for calculating basic mathematical operations on rectangle   
 * @author Dominik Stipić
 * @version 1.0
 */
public class Rectangle {
	
	/**
	 * Calculates area of rectangle for given dimensions by equation:
	 * Area = length * width.
	 * @param length Length of rectangle
	 * @param width Width of rectangle
	 * @return Area of rectangle as double
	 * @throws IllegalArgumentException - if the argumets are negative numbers or zero.
	 */
	public static double getArea(double length,double width) {
		if(length <= 0 || width <= 0 ) {
			throw new IllegalArgumentException("arguments can not be zero or negative numbers");
		}
		return length * width;
	}
	
	/**
	 * Calculates circumference of rectangle for given dimensions by equation:
	 * Circumference = 2*length + 2*width.
	 * @param length Length of rectangle
	 * @param width Width of rectangle
	 * @return Circumference of Rectangle
	 * @throws IllegalArgumentException - if the argumets are negative numbers or zero.
	 */
	public static double getCircumference(double length,double width) {
		if(length <= 0|| width <= 0 ) {
			throw new IllegalArgumentException("arguments can not be zero or negative numbers");
		}
		return 2 * length + 2 * width;
	}
	
	/**
	 * Method which takes user input from keyboard and checks if the input is valid (non-negative).
	 * Method will insist on valid input
	 * @param scanner Scanner object which reades input from keyboard.
	 * @param message 
	 * @return Correct user input as double
	 * @throws NumberFormatException - if the input isn't number	
	 */
	public static double inputFromKeyboard(Scanner scanner) {
		String input = null;
		
		while(true) {
			try {
				input = scanner.next();
				double number = Double.parseDouble(input);
				
				if(number <= 0) {
					System.out.println("Number can't be negative or zero,try again");
				}
				else {
					return number;
				}
				
			} catch (NumberFormatException e) {
				System.out.println(input + " can not be interpreted as number,try again");
			}
		}
	}
	
	/**
	 * Method parses and checks if the input is non-negative double
	 * @param arg the string which will be parsed to Double.
	 * @return The double value of arg.
	 * @throws IllegalArgumentException - if the string can not be parsed to double or if it is negative number.
	 * Exception holds appropriate error message.
	 */
	public static double parseInput(String arg) {
			try {
				double num = Double.parseDouble(arg);
				if(num < 0) {
					throw new IllegalArgumentException("number can not be negative");
				}
				
				return num;
				
			} catch (NumberFormatException e) {
				throw new IllegalArgumentException(arg + " can not be interpreted as number");
			}
	}
	
	/**
	 * Method which is automatically called when a program starts.
	 * @param args Arguments from command-line interface
	 */
	public static void main(String[] args) {
		double length = 0.0;
		double width = 0.0;
		
		if(args.length != 0) {			//We have arguments from CMD
			if(args.length == 2) {		//Correct number of arguments
				try {
					length = parseInput(args[0]);
					width = parseInput(args[1]);
					
				} catch (IllegalArgumentException e) {
					System.out.println(e.getMessage());
					System.exit(2);
				}
			}							//Wrong number of arguments
			else {		
				System.out.println("Unallowed number of arguments");
				System.exit(1);
			}
		}
		else {												//Arguments from Scanner
			try( Scanner scanner = new Scanner (System.in) ) {	//try-with-resources
				
				System.out.println("Enter rectangle length >");
				length = inputFromKeyboard(scanner);
				
				System.out.println("Enter rectangle width >");
				width = inputFromKeyboard(scanner);
			}
		}
		
		double area = getArea(length, width); 
		double circumference = getCircumference(length, width);
		
		System.out.format("Rectangle of length %.2f and width %.2f has area of %.2f and circumference of %.2f\n",length,width,area,circumference);
	}
	
}
