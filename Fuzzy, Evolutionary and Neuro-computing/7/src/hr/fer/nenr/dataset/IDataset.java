package hr.fer.nenr.dataset;

public interface IDataset<T> extends Iterable<T>{
	T get(int i);
	int size();
	IDataset<T> sample(int size);
	IDataset<T>[] split(double ratio);
}
