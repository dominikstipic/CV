package hr.fer.tel.rassus.main;

import java.util.ArrayList;
import java.util.List;

public class Main {

	public static List<Double> linspace(double upper, int n) {
		double delta = upper/n;
		List<Double> deltas = new ArrayList<>();
		for(int i = 1; i <= n; ++i) {
			double x = delta*i;
			deltas.add(x);
		}
		return deltas;
	}
	
	public static void main(String[] args) {
		System.out.println(linspace(6, 10));
		System.out.println(linspace(6, 10).size());
	}
	
}
