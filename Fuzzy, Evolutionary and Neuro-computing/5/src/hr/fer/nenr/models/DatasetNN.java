package hr.fer.nenr.models;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import hr.fer.nenr.interfaces.MLDataset;

public class DatasetNN implements MLDataset<Example>{
	private List<Example> dataset = new ArrayList<>();

	@Override
	public Iterator<Example> iterator() {
		return null;
	}

	@Override
	public void add(Example example) {
		
	}

	@Override
	public Example get(int idx) {
		return dataset.get(idx);
	}

	@Override
	public void remove(Example gesture) {
		
	}

	@Override
	public void remove(int idx) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public int getLabelCount(Example label) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void saveData(Example label) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public MLDataset<Example> sample(int size) {
		// TODO Auto-generated method stub
		return null;
	}

}
