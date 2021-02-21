package hr.fer.nenr.models;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class Repo {
	public static final Path ROOT = Paths.get("./repo");
	
	private static List<Integer> getList(){
		List<Integer> list = null;
		try {
			list = Files.list(ROOT).
                   	     map(p -> Integer.parseInt(p.getFileName().toString())).
                         collect(Collectors.toList());
		} catch (IOException e) {
			e.printStackTrace();
		}
		return list;
	}
	
	public static Path getNext() {
		Path path = ROOT;
		List<Integer> list = getList();
		if(list.isEmpty()) 
			path = path.resolve("1");
		else {
			Integer max = Collections.max(list) + 1;
			path = path.resolve(max.toString());
		}
		return path;
	}
	
	public static Path getLatest() {
		Path path = ROOT;
		List<Integer> list = getList();
		if(list.isEmpty()) 
			throw new IllegalStateException("Repo is empty");
		else {
			Integer max = Collections.max(list);
			path = path.resolve(max.toString());
		}
		return path;
	}
}
