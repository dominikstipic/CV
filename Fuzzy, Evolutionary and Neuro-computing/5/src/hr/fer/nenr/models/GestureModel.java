package hr.fer.nenr.models;

import java.awt.Point;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

import hr.fer.nenr.utils.Label;
import hr.fer.nenr.utils.PointUtils;

public class GestureModel implements Iterable<DPoint>{
	private List<DPoint> originalPoints = new ArrayList<>();
	public Label label;
	
	public GestureModel() {
	}
	
	public GestureModel(List<Point> points) {
		originalPoints = points.stream().map(p -> DPoint.fromPoint(p)).collect(Collectors.toList());
	}
	
	public void setOriginalPoints(List<DPoint> originalPoints) {
		this.originalPoints = originalPoints;
	}

	public void add(Point p) {
		DPoint dp = DPoint.fromPoint(p);
		originalPoints.add(dp);
	}
	
	public void add(DPoint dp) {
		originalPoints.add(dp);
	}
	
	public DPoint get(int idx) {
		return originalPoints.get(idx);
	}
	
	public void delete() {
		originalPoints = new ArrayList<>();
	}

	public int size() {
		return originalPoints.size();
	}
	
	public List<DPoint> getOriginalPoints() {
		return originalPoints;
	}
	
	public double lineDistance(DPoint target) {
		if(!originalPoints.contains(target)) throw new IllegalAccessError("DPoint isn't in gesture");
		DPoint current = originalPoints.get(0);
		double distance = 0;
		if(target.equals(current)) return 0;
		for(int i = 1; i < size(); ++i) {
			DPoint point = originalPoints.get(i);
			distance += PointUtils.euclidanDistance(current, point);
			if(point.equals(target)) break;
			current = point;
		}
		return distance;
	}

	@Override
	public Iterator<DPoint> iterator() {
		return originalPoints.iterator();
	}
	
	public GestureModel copy() {
		List<DPoint> p = new ArrayList<>();
		p.addAll(originalPoints);
		GestureModel model = new GestureModel();
		model.setOriginalPoints(p);
		return model;
	}
	
	public static GestureModel fromFile(Path path) {
		if(!Files.exists(path)) throw new IllegalArgumentException("File not found");
		if(!Files.isRegularFile(path)) throw new IllegalArgumentException("Path doesnt show to file");
		GestureModel model = new GestureModel();
		try(BufferedReader reader = Files.newBufferedReader(path)){
			List<String> lines = reader.lines().collect(Collectors.toList());
			for(String line : lines) {
				String[] str = line.split(",");
				double x = Double.valueOf(str[0]);
				double y = Double.valueOf(str[1]);
				DPoint point = new DPoint(x, y);
				model.add(point);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return model;
	}
	
	public List<Double> features(){
		List<Double> fs = new ArrayList<>();
		for(DPoint point : this.originalPoints) {
			fs.add(point.x);
			fs.add(point.y);
		}
		return fs;
	}

	@Override
	public String toString() {
		return originalPoints.toString();
	}

	@Override
	public int hashCode() {
		return originalPoints.hashCode();
	}

	@Override
	public boolean equals(Object obj) {
		if(!obj.getClass().equals(originalPoints.getClass())) return false;
		GestureModel other = (GestureModel) obj;
		return other.getOriginalPoints().equals(originalPoints);
	}
	
	
	
	
}
