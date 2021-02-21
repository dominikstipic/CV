package hr.fer.nenr.models;

import java.util.Arrays;

public class Data {
	public final double[] example;
	public final double[] oneHot;
	
	public Data(double[] example, double[] oneHot) {
		this.example = example;
		this.oneHot = oneHot;
	}

	public int classId() {
		int label = 0;
		for(int i = 0; i < oneHot.length; ++i) {
			if(oneHot[i] == 1) {
				label = i;
				break;
			}
		}
		return label;
	}
	
	@Override
	public String toString() {
		String ex = Arrays.toString(example);
		int oh = classId();
		return ex + ", " + oh;
	}
	
	
}
