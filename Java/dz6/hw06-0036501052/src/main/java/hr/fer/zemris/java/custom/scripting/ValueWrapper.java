package hr.fer.zemris.java.custom.scripting;

import java.util.Set;

/**
 * Holds the Object value and provides simple arithmetic and logic operations which 
 * can be used. Operations are: adding,subtraction,multiplication,divison and comaparison.
 * Operations can be applied just to : Integer,Double and Strings which can be parsed to Number.
 *  
 * @author Dominik Stipic 
 *
 */
public class ValueWrapper {
	/**
	 * Stored value
	 */
	private Object value;
	/**
	 * Allowed data in operations
	 */
	private Set<Class<?>> allowedOperationTypes = Set.of(Integer.class, String.class, Double.class);

	/**
	 * Constructs ValueWrapper which holds provided value
	 * @param value which is going to be stored
	 */
	public ValueWrapper(Object value) {
		this.value = value;
	}

	/**
	 * Adds given value to stored value
	 * @param incValue value which is going to be added to stored value
	 * @throws IllegalArgumentException - if given value is not:Double,Integer or 
	 * String which can be parsed to Number
	 */
	public void add(Object incValue) {
		value = performOperation(value, incValue ,Operation.ADD);
	}
	
	/**
	 * Subtracts stored value with given one
	 * @param decValue value which is going to subtract stored value
	 * @throws IllegalArgumentException - if given value is not:Double,Integer or 
	 * String which can be parsed to Number
	 */
 	public void subtract(Object decValue) {
 		value = performOperation(value, decValue, Operation.SUBSTRACT);
 	}
	
 	/**
	 * Multiplies stored value with given one
	 * @param mulValue value which is going to multiply stored value
	 * @throws IllegalArgumentException - if given value is not:Double,Integer or 
	 * String which can be parsed to Number
	 */
 	public void multiply(Object mulValue) {
 		value = performOperation(value, mulValue, Operation.MULTIPLY);
 	}
 	
 	/**
	 * Divides stored value with given one
	 * @param divValue value which is going to divide stored value
	 * @throws IllegalArgumentException - if given value is not:Double,Integer or 
	 * String which can be parsed to Number
	 * @throws ArithmeticException - if divison with zero occured
	 */
 	public void divide(Object divValue) {
 		value = performOperation(value, divValue, Operation.DIVIDE);
 	}
 	
 	/**
	 * Compares two values 
	 * @param value which is going to be compared with stored value
	 * @throws IllegalArgumentException - if given value is not:Double,Integer or 
	 * String which can be parsed to Number
	 */
 	public int numCompare(Object withValue) {
 		return ((Number)performOperation(value, withValue, Operation.COMPARE)).intValue();
 	}
 	
	/**
	 * Checks if provided value satisfy all needed conditions for operation process
	 * @param otherValue
	 */
	private void checkType(Object otherValue) {
		if(!(otherValue == null || allowedOperationTypes.contains(otherValue.getClass()))){
			throw new RuntimeException("allowed data types for operation are : null, Double, Integer, String.You provided :" + otherValue.getClass());
		}
		if(otherValue != null && otherValue.getClass().equals(String.class)) {
			String str = ((String) otherValue).trim();
			if(!str.matches("-?[0-9]+(.[0-9]+)?(E-?[0-9]+)?")) {
				throw new IllegalArgumentException("Provided string cannot be parsed to double or integer value: " + str);
			}
		}
	}

	/**
	 * Gets the stored value
	 * @return stored value
	 */
	public Object getValue() {
		return value;
	}
	
	/**
	 * Sets the stored value
	 * @param value new value which is going to be stored
	 */
	public void setValue(Object value) {
		this.value = value;
	}
	
	/**
	 * Performes operation and checks all necessary things.
	 * @param value1 
	 * @param value2
	 * @param type of operation
	 * @return result of operation
	 * @throws IllegalArgumentException - if the incorrect data type is passed
	 * @throws ArithmeticException - if divison with zero occured
	 */
	private Object performOperation(Object value1, Object value2, Operation type) {
		checkType(value1);
		checkType(value2);
		
		if(value1 == null) {
			value1 = 0;
		}
		if(value2 == null) {
			value2 = 0;
		}
		if(value1.getClass().equals(String.class)) {
			value1 = parseString(value1);
		}
		if(value2.getClass().equals(String.class)) {
			value2 = parseString(value2);
		}
	
		Number num1 = (Number) value1;
		Number num2 = (Number) value2;
		
		if(value1.getClass().equals(Integer.class) && value2.getClass().equals(Integer.class)) {
			return intOperation(num1.intValue(), num2.intValue(), type);
		}
		
		return doubleOperation(num1.doubleValue(), num2.doubleValue(), type);
	}
	
	/**
	 * Performs operation with Integers and returns one
	 * @param i1 Double value
	 * @param i2 Double value 
	 * @param type of operation
	 * @return result of operation
	 * @throws ArithmeticException - if divison with zero occured
	 */
	private int intOperation (Integer i1, Integer i2, Operation type) {
		if(type.equals(Operation.ADD)) {
			return i1 + i2;
		}
		else if (type.equals(Operation.SUBSTRACT)) {
			return i1 - i2;
		}
		else if (type.equals(Operation.MULTIPLY)) {
			return i1 * i2;
		}
		else if(type.equals(Operation.DIVIDE)){
			if(i2 == 0) {
				throw new ArithmeticException("Can not divide by zero");
			}
			return i1/i2;
		}
		else {
			return i1.compareTo(i2);
		}
	}
	
	/**
	 * Performs operation with doubles and returns one
	 * @param d1 Double value
	 * @param d2 Double value 
	 * @param type of operation
	 * @return result of operation
	 * @throws ArithmeticException - if divison with zero occured
	 */
	private double doubleOperation (Double d1, Double d2, Operation type) {
		if(type.equals(Operation.ADD)) {
			return d1 + d2;
		}
		else if (type.equals(Operation.SUBSTRACT)) {
			return d1 - d2;
		}
		else if (type.equals(Operation.MULTIPLY)) {
			return d1 * d2;
		}
		else if(type.equals(Operation.DIVIDE)){
			if(d2 == 0) {
				throw new ArithmeticException("Can not divide by zero");
			}
			return d1/d2;
		}
		else {
			return d1.compareTo(d2);
		}
	}
	
	/**
	 * Parses string to Double or 
	 * Integer
	 * @param obj String which is going to be parsed
	 * @return number which is extracted from given string
	 */
	private Number parseString(Object obj) {
		String str = (String) obj;
		if(str.matches("[0-9]+")) {
			return Integer.parseInt(str);
		}
		else {
			return Double.parseDouble(str);
		}
	}
	
	/**
	 * Allowed operations 
	 * @author Dominik Stipic
	 *
	 */
	private enum Operation{
		ADD,
		SUBSTRACT,
		MULTIPLY,
		DIVIDE,
		COMPARE
	}
	
}
