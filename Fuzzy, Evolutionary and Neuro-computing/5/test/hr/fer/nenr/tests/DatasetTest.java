package hr.fer.nenr.tests;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.junit.Test;

import hr.fer.nenr.interfaces.MLDataset;
import hr.fer.nenr.models.DPoint;
import hr.fer.nenr.models.GestureModel;
import hr.fer.nenr.nn.NeuralDataset;
import hr.fer.nenr.utils.Label;

public class DatasetTest {
	
	public static List<Path> getFiles(Path dirPath) {
		List<Path> xs = null;
		try (Stream<Path> paths = Files.walk(dirPath)) {
		    xs = paths.filter(Files::isRegularFile).collect(Collectors.toList());
		} catch (IOException e) {
			e.printStackTrace();
		} 
		return xs;
	}
	
	public static List<GestureModel> fun(){
		String dir = NeuralDataset.PATH.toString();
		List<Label> labels = List.of(Label.ALPHA);
		List<GestureModel> gestures = new ArrayList<>();
		for(Label label : labels) {
			String dirLabelName = dir + "/" + label.label;
			Path dirPath = Paths.get(dirLabelName);
			if(!Files.exists(dirPath)) continue;
			List<Path> files = getFiles(dirPath);
			for(Path filePath : files) {
				try(BufferedReader reader = Files.newBufferedReader(filePath)){
					List<String> lines = reader.lines().collect(Collectors.toList());
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
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return gestures;
	}
	
	@Test
	public void nnTest1() {
		Path path = NeuralDataset.PATH;
		MLDataset dataset = new NeuralDataset(path, 20);
		System.out.println(dataset.size());
		
//		List<GestureModel> models = fun();
//		for(GestureModel m : models) {
//			for(int i = 1; i < m.size();++i) {
//				GestureModel m1 = Preprocess.preprocess(m, i);
//				if(m1.size() != i) System.out.println(m1.size() + "," + i);
//				
//			}
//		}
//		
//		System.out.println("Ok");
	}
	
	
	
}
