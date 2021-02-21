package hr.fer.nenr.models;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import hr.fer.nenr.interfaces.Dataset;
import hr.fer.nenr.utils.Label;

public class GestureDataset implements Dataset<GestureModel, Label>{
	private Map<GestureModel, Label> examples = new HashMap<>();
	public static final String PATH = "./database";
	
	@Override
	public Iterator<Entry<GestureModel, Label>> iterator() {
		Iterator<Entry<GestureModel, Label>> iter = examples.entrySet().iterator();
		return iter;
	}

	@Override
	public void add(GestureModel gesture, Label label) {
		examples.put(gesture, label);
	}

	@Override
	public Entry<GestureModel, Label> get(int idx) {
		int k = 0;
		Entry<GestureModel, Label> e = null;
		for(Entry<GestureModel, Label> entry : examples.entrySet()) {
			if(k == idx) {
				e = entry;
				break;
			}
			++k;
		}
		return e;
	}

	@Override
	public void remove(GestureModel gesture) {
		examples.remove(gesture);
	}

	@Override
	public void remove(int idx) {
		GestureModel key = get(idx).getKey();
		remove(key);
	}

	@Override
	public int getLabelCount(Label label) {
		int count = 0;
		for(Entry<GestureModel, Label> e : examples.entrySet()) {
			Label lab = e.getValue();
			if(lab.equals(label))++count;
		}
		return count;
	}

	private List<GestureModel> byValue(Label label){
		List<GestureModel> set = examples.entrySet().
				                         stream().
				                         filter(e -> e.getValue().equals(label)).
				                         map(e -> e.getKey()).
				                         collect(Collectors.toList());
		return set;
	}
	
	private void createPath(Path path, final boolean isFile) {
		Consumer<Path> create = p -> {
			try {
				if(isFile) {
					Files.deleteIfExists(p);
					Files.createFile(p);
				}
				else {
					if(!Files.exists(path)) Files.createDirectory(p);	
				}
			}
			catch (Exception e) {
				e.printStackTrace();
			}
		};
		create.accept(path);
	}
	
	@Override
	public void saveData(Label label) {
		List<GestureModel> gestures = byValue(label);
		String labelName = label.name();
		String dirName = PATH + "/" + labelName;
		Path dirPath = Paths.get(dirName);
		createPath(dirPath, false);
		for(int i = 0; i < gestures.size(); ++i){
			String fileName = dirName + "/" + i + ".txt";
			Path filePath = Paths.get(fileName);
			createPath(filePath, true);
			try(BufferedWriter writer = Files.newBufferedWriter(filePath)){
				GestureModel model = gestures.get(i);
				for(DPoint point : model) {
					String line = point.x + "," + point.y;
					writer.append(line);
					writer.newLine();
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return;
	}
	
	

}
