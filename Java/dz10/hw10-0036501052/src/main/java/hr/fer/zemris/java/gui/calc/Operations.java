package hr.fer.zemris.java.gui.calc;

import static java.lang.Math.acos;
import static java.lang.Math.asin;
import static java.lang.Math.atan;
import static java.lang.Math.cos;
import static java.lang.Math.log;
import static java.lang.Math.log10;
import static java.lang.Math.pow;
import static java.lang.Math.sin;
import static java.lang.Math.tan;

import java.util.HashMap;
import java.util.Map;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;

/**
 * Utility class for operations
 * @author DominikStipić
 *
 */
public class Operations {
	
	//ALL NECESSARY OPERATIONS
	public static DoubleBinaryOperator ADD = (v1,v2) -> v1 + v2;
	public static DoubleBinaryOperator SUB = (v1,v2) -> v1 - v2;
	public static DoubleBinaryOperator MUL = (v1,v2) -> v1 * v2;
	public static DoubleBinaryOperator DIV = (v1,v2) -> v1 / v2;
	public static DoubleBinaryOperator ROOT_N = (v,n) -> pow(v,1/n);
	public static DoubleBinaryOperator POW_N = (v,n) -> pow(v,n);

	public static DoubleUnaryOperator INV = v -> 1/v;
	private static DoubleUnaryOperator LOG = v -> log10(v);
	private static DoubleUnaryOperator LN = v -> log(v);
	private static DoubleUnaryOperator SIN = v -> sin(v);
	private static DoubleUnaryOperator COS = v -> cos(v);
	private static DoubleUnaryOperator TAN = v -> tan(v);
	private static DoubleUnaryOperator CTG = v -> cos(v)/sin(v);
	
	private static DoubleUnaryOperator POW_10 = v -> pow(10,v);
	private static DoubleUnaryOperator E = v -> pow(Math.E, v);
	private static DoubleUnaryOperator ASIN = v -> asin(v);
	private static DoubleUnaryOperator ACOS = v -> acos(v);
	private static DoubleUnaryOperator ATAN = v -> atan(v);
	private static DoubleUnaryOperator ACTG = v -> 1/atan(v);

	/**
	 * Map which asscoiates  unary operation with their name
	 */
	private static Map<String,DoubleUnaryOperator> map = fillData();
	
	/**
	 * fills the map
	 * @return filled map
	 */
	private static Map<String,DoubleUnaryOperator> fillData() {
		 Map<String,DoubleUnaryOperator> map = new HashMap<>();
	 	map.put("log",LOG);
		map.put("ln",LN);
		map.put("sin",SIN);
		map.put("cos",COS);
		map.put("tan",TAN);
		map.put("ctg",CTG);
		map.put("10^",POW_10);
		map.put("e^",E);
		map.put("asin",ASIN);
		map.put("acos",ACOS);
		map.put("acos",ACOS);
		map.put("atan",ATAN);
		map.put("actg",ACTG);
		map.put("1/x", INV);
		return map;
	}
	
	/**
	 * static variable which corresponds to inverse or normal unary operations
	 */
	public static boolean inv = false;
	
	/**
	 * changes to differed mode of operations
	 */
	public static void changeMode() {
		Operations.inv = !Operations.inv;
	}
	
	//suppliers for unary operation names
	public static final Supplier<String> GET_SIN = () -> inv ? "asin" : "sin";
	public static final Supplier<String> GET_COS = () -> inv ? "acos" : "cos";
	public static final Supplier<String> GET_TAN = () -> inv ? "atan" : "tan";
	public static final Supplier<String> GET_CTG = () -> inv ? "actg" : "ctg";
	public static final Supplier<String> GET_LOG = () -> inv ? "10^" : "log"; 
	public static final Supplier<String> GET_LN = () -> inv ? "e^" : "ln";
	public static final Supplier<String> GET_NPOW = () -> inv ? "x^(1/n)" : "x^n";
	public static final Supplier<String> GET_INV = () -> "1/x";
	
	/**
	 * gets the unary operation from this class map
	 * @param key of unary operation
	 * @return operation
	 */
	public static DoubleUnaryOperator getUnaryOperation (String key) {
		return map.get(key);
	}
}
