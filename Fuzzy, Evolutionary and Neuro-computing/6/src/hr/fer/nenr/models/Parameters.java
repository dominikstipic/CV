package hr.fer.nenr.models;

import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;

import Jama.Matrix;
import hr.fer.nenr.utils.MatrixAdapter;

public class Parameters implements Iterable<Entry<String, Matrix>>{
	private HashMap<String, Matrix> paramsMap   = new HashMap<>();
	private HashMap<String, Matrix> paramsGrads = new HashMap<>();
	
	public Parameters(List<String> names, List<Matrix> params) {
		if(names.size() != params.size()) throw new IllegalArgumentException("Sizes dont match");
		for(int i = 0; i < names.size(); ++i) {
			String key = names.get(i); Matrix value = params.get(i);
			paramsMap.put(key, value);
		}
	}
	
	
	private Parameters(HashMap<String, Matrix> paramsMap, HashMap<String, Matrix> paramsGrads) {
		this.paramsMap = paramsMap;
		this.paramsGrads = paramsGrads;
	}



	@Override
	public Iterator<Entry<String, Matrix>> iterator() {
		return paramsMap.entrySet().iterator();
	}
	
	public Matrix getParam(String key) {
		Matrix param = paramsMap.get(key);
		return param;
	}
	
	public Collection<String> getKeySet(){
		return paramsMap.keySet();
	}
	
	public void addParam(String key, Matrix value) {
		paramsMap.put(key, value);
	}
	
	public Parameters copy() {
	    HashMap<String, Matrix> params = new HashMap<>();
	    HashMap<String, Matrix> grads = new HashMap<>();
	    paramsMap.entrySet().forEach(e -> params.put(e.getKey(), e.getValue()));
	    paramsGrads.entrySet().forEach(e -> grads.put(e.getKey(), e.getValue()));
	    return new Parameters(params, grads);
	    
	}
	
	public Matrix getGrad(String key) {
		Matrix grad = paramsGrads.get(key);
		return grad;
	}
	
	public void addGrads(String key, Matrix grads) {
		if(!paramsMap.containsKey(key)) throw new IllegalArgumentException("Parameter doesn't exist");
		Matrix param = paramsMap.get(key);
		if(!MatrixAdapter.checkDim(grads, param)) throw new IllegalArgumentException("Gradient and parameter dimensions dont match");
		paramsGrads.put(key, grads);
	}
	
	public void addGrads(List<String> keys, List<Matrix> grads) {
		if(keys.size() != grads.size()) 
			throw new IllegalArgumentException("Array sizes arent the same");
		for(int i = 0; i < keys.size(); ++i) {
			addGrads(keys.get(i), grads.get(i));
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for(Entry<String, Matrix> e : paramsMap.entrySet()) {
			String key = e.getKey(); Matrix m = e.getValue();
			String s = String.format("%s = %s\n", key, MatrixAdapter.matrixString(m));
			sb.append(s);
		}
		return sb.toString();
	}
	
	public String gradString() {
		StringBuilder sb = new StringBuilder();
		for(Entry<String, Matrix> e : this.paramsGrads.entrySet()) {
			String key = e.getKey(); Matrix m = e.getValue();
			String s = String.format("D%s = %s\n", key, MatrixAdapter.matrixString(m));
			sb.append(s);
		}
		return sb.toString();
	}
	
}
