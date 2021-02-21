package hr.fer.nenr.nn;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import hr.fer.nenr.interfaces.MLDataset;
import hr.fer.nenr.models.DPoint;
import hr.fer.nenr.models.Example;
import hr.fer.nenr.models.GestureModel;
import hr.fer.nenr.utils.Label;
import hr.fer.nenr.utils.Preprocess;
import hr.fer.nenr.utils.VectorUtils;

public class NeuralDataset implements MLDataset{
	private List<Example> examples = new ArrayList<>();
	public static final Path PATH = Paths.get("./database");
	
	public NeuralDataset(Path path, int M) {
		if(!Files.exists(path))
			throw new IllegalArgumentException("Path doesn't exist");
		List<GestureModel> models = fromPath(path);
		for(GestureModel model : models) {
			GestureModel processed = Preprocess.preprocess(model, M);
			List<Double> features = processed.features();
			Example ex = new Example(features, model.label.value);
			examples.add(ex);
		}
	}
	
	public NeuralDataset(List<Example> data) {
		this.examples = data;
	}

	@Override
	public Iterator<Example> iterator() {
		return examples.iterator();
	}

	@Override
	public void add(Example example) {
		examples.add(example);
	}

	@Override
	public Example get(int idx) {
		return examples.get(idx);
	}

	@Override
	public void remove(Example gesture) {
		examples.remove(gesture);
	}

	@Override
	public void remove(int idx) {
		examples.remove(idx);
	}

	
	
	@Override
	public MLDataset sample(int size) {
		List<Example> sample = new ArrayList<>();
		List<Integer> samplesIdx = VectorUtils.randomVector(this.size(), size);
		samplesIdx.forEach(idx -> sample.add(get(idx)));
		MLDataset dataset = new NeuralDataset(sample);
		return dataset;
	}

	
	private List<Path> getFiles(Path dirPath) {
		List<Path> xs = null;
		try (Stream<Path> paths = Files.walk(dirPath)) {
		    xs = paths.filter(Files::isRegularFile).collect(Collectors.toList());
		} catch (IOException e) {
			e.printStackTrace();
		} 
		return xs;
	}
	
	private List<String> readLines(Path path){
		List<String> lines = null;
		try(BufferedReader reader = Files.newBufferedReader(path)){
			lines = reader.lines().collect(Collectors.toList());
		} catch (IOException e) {
			e.printStackTrace();
		}
		return lines;
	}
	
	private List<GestureModel> fromPath(Path path){
		String dir = path.toString();
		List<Label> labels = List.of(Label.ALPHA,
								     Label.BETA,
								     Label.DELTA, 
								     Label.ETHA, 
								     Label.GAMMA);
		List<GestureModel> gestures = new ArrayList<>();
		for(Label label : labels) {
			String dirLabelName = dir + "/" + label.label;
			Path dirPath = Paths.get(dirLabelName);
			if(!Files.exists(dirPath)) continue;
			List<Path> files = getFiles(dirPath);
			for(Path filePath : files) {
				List<String> lines = readLines(filePath);
				GestureModel model = new GestureModel();
				for(String line : lines) {
					String[] arr = line.split(",");
					double a = Double.valueOf(arr[0]);
					double b = Double.valueOf(arr[1]);
					DPoint point = new DPoint(a, b);
					model.add(point);
				}
				model.label = label;
				gestures.add(model);
				 
			}
		}
		return gestures;
	}
	
	public MLDataset getExamples(Integer ...ints) {
		List<Example> examples = new ArrayList<>();
		for(int idx : ints) {
			examples.add(this.get(idx));
		}
		return new NeuralDataset(examples);
	}

	@Override
	public int size() {
		return examples.size();
	}

	@Override
	public String toString() {
		return examples.toString();
	}


	@Override
	public MLDataset[] split(double ratio) {
		int train_size = (int) (size() * ratio);
		List<Example> valid = new ArrayList<>(examples);
		Collections.shuffle(valid);
		List<Example> train = new ArrayList<>();
		
		for(int i = 0; i < train_size; ++i) {
			Example ex = valid.get(0);
			train.add(ex);
			valid.remove(0);
		}
		NeuralDataset x = new NeuralDataset(train);
		NeuralDataset y = new NeuralDataset(valid);
		return new NeuralDataset[]{x, y};
	}
	
	
}
