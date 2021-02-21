package hr.fer.nenr.dataset;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import hr.fer.nenr.models.Data;
import hr.fer.nenr.utils.Utils;

public class Dataset implements IDataset<Data>, Iterable<Data>{
	private static final String PATH = "./data/data.csv"; 
	private List<Data> data = new ArrayList<>();
	
	private Dataset(List<Data> data) {
		this.data = data;
	}

	public static IDataset<Data> build() {
		List<Data> data = Utils.readCSV(PATH);
		IDataset<Data> dataset = new Dataset(data);
		return dataset;
	}
	
	@Override
	public Data get(int i) {
		return data.get(i);
	}

	@Override
	public int size() {
		return data.size();
	}

	@Override
	public Iterator<Data> iterator() {
		return data.iterator();
	}

	@Override
	public IDataset<Data>[] split(double ratio) {
		int train_size = (int) (size() * ratio);
		List<Data> valid = new ArrayList<>(data);
		Collections.shuffle(valid);
		List<Data> train = new ArrayList<>();
		for(int i = 0; i < train_size; ++i) {
			Data ex = valid.get(0);
			train.add(ex);
			valid.remove(0);
		}
		Dataset x = new Dataset(train);
		Dataset y = new Dataset(valid);
		return new Dataset[]{x, y};
	}

	@Override
	public IDataset<Data> sample(int size) {
		List<Data> sample = new ArrayList<>();
		List<Integer> samplesIdx = Utils.randomVector(this.size(), size);
		samplesIdx.forEach(idx -> sample.add(get(idx)));
		IDataset<Data> dataset = new Dataset(sample);
		return dataset;
	}

}
