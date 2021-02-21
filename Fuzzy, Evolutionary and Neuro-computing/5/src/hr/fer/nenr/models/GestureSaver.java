package hr.fer.nenr.models;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

import hr.fer.nenr.utils.Label;

public class GestureSaver{
	public static final Path PATH = Paths.get("./database");
	private int cnt = 0;
	private Map<Label, Integer> map = new HashMap<>();
	
	public GestureSaver() {
		try {
			if(Files.exists(PATH)) {
				deleteDir();
			}
			Files.createDirectory(PATH);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private void deleteDir() throws IOException {
		Files.walkFileTree(PATH, new SimpleFileVisitor<Path>() {
			   @Override
			   public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
			       Files.delete(file);
			       return FileVisitResult.CONTINUE;
			   }

			   @Override
			   public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
			       Files.delete(dir);
			       return FileVisitResult.CONTINUE;
			   }
			});
	}
	
	private String getString() {
		Long time = System.nanoTime();
		return String.valueOf(time);
	}
	
	private void updateData(Label label) {
		++cnt;
		if(map.containsKey(label)) {
			int i = map.get(label) + 1;
			map.put(label, i);
		}
		else {
			map.put(label, 1);
		}
	}
	
	public int getLabelCount(Label label) {
		if(map.containsKey(label)) return map.get(label);
		else return 0;
	}
	
	public void save(GestureModel gesture, Label label) {
		updateData(label);
		String labelName = label.name();
		String dirName = PATH + "/" + labelName;
		Path dirPath = Paths.get(dirName);
		createPath(dirPath, false);
		String fileName = dirName + "/" + getString() + ".txt";
		Path filePath = Paths.get(fileName);
		createPath(filePath, true);
		try(BufferedWriter writer = Files.newBufferedWriter(filePath)){
			for(DPoint point : gesture) {
				String line = point.x + "," + point.y;
				writer.write(line);
				writer.newLine();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
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

	public int getCnt() {
		return cnt;
	}

	

}
