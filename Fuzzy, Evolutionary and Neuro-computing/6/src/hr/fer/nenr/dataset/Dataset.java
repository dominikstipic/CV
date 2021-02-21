package hr.fer.nenr.dataset;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.function.BiFunction;

import hr.fer.nenr.models.Example;
import hr.fer.nenr.utils.VectorUtils;

public class Dataset implements MLDataset{
	private List<Example> data = new ArrayList<>();

	public Dataset(List<Example> data) {
		this.data = data;
	}

	@Override
	public Iterator<Example> iterator() {
		return data.iterator();
	}

	@Override
	public void add(Example example) {
		data.add(example);
	}

	@Override
	public Example get(int idx) {
		return data.get(idx);
	}

	@Override
	public void remove(Example example) {
		data.remove(example);
	}

	@Override
	public void remove(int idx) {
		data.remove(idx);
	}
	
	@Override
	public int size() {
		return data.size();
	}

	@Override
	public MLDataset sample(int size) {
		List<Example> sample = new ArrayList<>();
		List<Integer> samplesIdx = VectorUtils.randomVector(this.size(), size);
		samplesIdx.forEach(idx -> sample.add(get(idx)));
		MLDataset dataset = new Dataset(sample);
		return dataset;
	}

	@Override
	public MLDataset[] split(double ratio) {
		int train_size = (int) (size() * ratio);
		List<Example> valid = new ArrayList<>(data);
		Collections.shuffle(valid);
		List<Example> train = new ArrayList<>();
		for(int i = 0; i < train_size; ++i) {
			Example ex = valid.get(0);
			train.add(ex);
			valid.remove(0);
		}
		Dataset x = new Dataset(train);
		Dataset y = new Dataset(valid);
		return new Dataset[]{x, y};
	}
	
	public static MLDataset sampleFunction(int m, BiFunction<Double, Double, Double> function) {
		int from = -m;
		int samples = 2*m + 1;
		List<Example> examples = new ArrayList<>();
		for(int i = 0; i < samples; ++i) {
			double x = from + i;
			for(int j = 0; j < samples; ++j) {
				double y = from + j;
				double target = function.apply(x, y);
				Example e = new Example(Arrays.asList(x,y), Arrays.asList(target));
				examples.add(e);
			}
		}
		return new Dataset(examples);
	}

	
}
