package hr.fer.nenr.dataset;

import hr.fer.nenr.models.Example;

public interface MLDataset extends Iterable<Example>{
	void add(Example example);
	Example get(int idx);
	void remove(Example example);
	void remove(int idx);
	MLDataset sample(int size);
	int size();
	MLDataset[] split(double ratio);
}
