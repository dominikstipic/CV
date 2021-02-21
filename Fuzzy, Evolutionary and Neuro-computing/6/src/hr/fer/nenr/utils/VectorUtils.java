package hr.fer.nenr.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class VectorUtils {

	public static <T> List<T> vectorUniFunction(List<T> vector, Function<T, T> function){
		List<T> list = vector.stream().map(d -> function.apply(d)).collect(Collectors.toList());
		return list;
	} 
	
	public static <T> List<T> vectorBiFunction(List<T> xs, List<T> ys, BiFunction<T, T, T> function){
		if(xs.size() != ys.size()) throw new IllegalArgumentException("Vector sizes doesnt match");
		List<T> result = new ArrayList<T>();
		for(int i = 0; i < xs.size(); ++i) {
			T value = function.apply(xs.get(i), ys.get(i));
			result.add(value);
		}
		return result;
	} 
	
	public static List<Double> mul(List<Double> xs, List<Double> ys){
		BiFunction<Double, Double, Double> function = (d1, d2) -> d1*d2;
		return vectorBiFunction(xs, ys, function);
	} 
	
	public static List<Double> scalarMul(List<Double> xs, double scalar){
		Function<Double, Double> function = d -> d*scalar;
		return vectorUniFunction(xs, function);
	} 
	
	public static List<Double> add(List<Double> xs, List<Double> ys){
		BiFunction<Double, Double, Double> function = (d1, d2) -> d1+d2;
		return vectorBiFunction(xs, ys, function);
	} 
	
	public static List<Double> minus(List<Double> xs, List<Double> ys){
		BiFunction<Double, Double, Double> function = (d1, d2) -> d1-d2;
		return vectorBiFunction(xs, ys, function);
	} 
	
	public static List<Double> power(List<Double> xs, int n){
		Function<Double, Double> function = d -> Math.pow(d, 2);
		return vectorUniFunction(xs, function);
	} 
	
	public static List<Integer> oneHot(int classId, int N){
		List<Integer> arr = replicate(0, N);
		arr.set(classId, 1);
		return arr;
	} 
	
	public static List<Double> sum(List<Double> xs){
		double sum = 0;
		for(Double d : xs) sum += d;
		return Arrays.asList(sum);
	} 
	
	public static Double vectorScalar(List<Double> xs, List<Double> ys){
		List<Double> result = mul(xs, ys);
		double r = sum(result).get(0);
		return r;
	} 
	
	public static <T> List<T> replicate(T x, int size){
		List<T> xs = new ArrayList<>();
		for(int i = 0; i < size; ++i) xs.add(x);
		return xs;
	} 
	
	public static <T> List<List<T>> matrixReplicate(T x, int rows, int cols){
		List<List<T>> X = new ArrayList<>();
		for(int i = 0; i < rows; ++i) {
			List<T> xs = new ArrayList<T>();
			for(int j = 0; j < cols; ++j) {
				xs.add(x);
			}
			X.add(xs);
		}
		return X;
	} 
	
	public static <T> void dimCheck(List<T> xs, int x) {
		if(xs.size() != x) throw new IllegalArgumentException("dimensions doesn't match");
	}
	
	public static <T, S> void dimCheck(List<T> xs, List<S> ys){
		dimCheck(xs, ys.size());
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
}
