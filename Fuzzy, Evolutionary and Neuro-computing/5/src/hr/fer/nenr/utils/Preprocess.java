package hr.fer.nenr.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import hr.fer.nenr.models.DPoint;
import hr.fer.nenr.models.GestureModel;
import static hr.fer.nenr.utils.PointUtils.*;

public class Preprocess {
	
	public static GestureModel mean(GestureModel original) {
		List<DPoint> originalPoints = original.getOriginalPoints();
		DPoint meanPoint = PointUtils.mean(originalPoints);
		List<DPoint> newPoints = new ArrayList<>();
		
		for(DPoint p : originalPoints) {
			double newX =  p.x - meanPoint.x;
			double newY =  p.y - meanPoint.y;
			DPoint point = new DPoint(newX, newY);
			newPoints.add(point);
		}
		GestureModel model = new GestureModel();
		model.setOriginalPoints(newPoints);
		return model;
	}
	
	public static GestureModel normalization(GestureModel original) {
		List<DPoint> originalPoints = original.getOriginalPoints();
		
		double maxX = 0;
		double maxY = 0;
		for(DPoint p : originalPoints) {
			maxX = Math.max(maxX, p.x);
			maxY = Math.max(maxY, p.y);
		}
		double m = Math.max(maxX, maxY);
		
		List<DPoint> newPoints = new ArrayList<>();
		for(DPoint p : originalPoints) {
			double newX = p.x/m;
			double newY = p.y/m;
			DPoint point = new DPoint(newX, newY);
			newPoints.add(point);
		}
		GestureModel model = new GestureModel();
		model.setOriginalPoints(newPoints);
		return model;
	}
	
	public static GestureModel applyFunction(GestureModel model, Function<DPoint, DPoint> function) {
		List<DPoint> originalPoints = model.getOriginalPoints();
		
		List<DPoint> newPoints = new ArrayList<>();
		for(DPoint p : originalPoints) {
			DPoint newPoint = function.apply(p);
			newPoints.add(newPoint);
		}
		GestureModel m = new GestureModel();
		m.setOriginalPoints(newPoints);
		return m;
	}
	
	public static GestureModel changeInterval(GestureModel model, int H, int W, int h, int w) {
		Function<DPoint, DPoint> transform = p -> {
			double x = (p.x * w)/W;
			double y = (p.y * h)/H;
			DPoint point = new DPoint(x, y);
			return point;
		};
		return applyFunction(model, transform);
	}
	
	public static double gestureLength(GestureModel model) {
		double distance = 0;
		for(int i = 0; i < model.size()-1; ++i) {
			DPoint first = model.get(i);
			DPoint second = model.get(i+1);
			distance += euclidanDistance(first, second);
		}
		return distance;
	}
	
	public static GestureModel sample(GestureModel model, int M) {
		if(M > model.size()) {
			M  = model.size();
		}
		else if(M < 1) M = 1;
		double D = gestureLength(model);
		GestureModel sampledModel = new GestureModel();
		if(M == 1) {
			sampledModel.add(model.get(0));
			return sampledModel;
		}
		for(int k = 0; k < M; ++k) {
			double radius = k*D/(M-1);
			for(int i = 0; i < model.size(); ++i) {
				DPoint current = model.get(i);
				double d = model.lineDistance(current);
				if(d > radius) {
					sampledModel.add(model.get(i-1));
					break;
				}
				else if(Double.compare(d, radius) == 0) {
					sampledModel.add(model.get(i));
					break;
				}
			}
		}
		if(sampledModel.size() != M) {
			sampledModel.add(model.get(model.size()-1));
		}
		return sampledModel;
	}
	
	public static GestureModel preprocess(GestureModel model, int M) {
		model = mean(model);
		model = normalization(model);
		model = sample(model, M);
		return model;
	}
}
