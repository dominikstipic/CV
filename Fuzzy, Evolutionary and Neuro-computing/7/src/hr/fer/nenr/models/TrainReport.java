package hr.fer.nenr.models;

import java.awt.Color;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.github.plot.Plot;
import com.github.plot.Plot.DataSeriesOptions;
import com.github.plot.Plot.Line;

import hr.fer.nenr.utils.LinAlg;
	

public class TrainReport {
	public List<Double> errors;
	private static Color[] colors = {Color.GREEN, Color.RED, Color.BLUE};
	private static Color supportColor = Color.ORANGE;
	
	public static void linePlot(List<Double> errors) {
		double max_error = Collections.max(errors);
		double min_error = Collections.min(errors);
		List<Double> generation = IntStream.range(0, errors.size()).boxed().mapToDouble(i -> (double)i).boxed().collect(Collectors.toList());
		Plot plot = Plot.plot(null).xAxis("X", Plot.axisOpts().range(0, generation.size())).
				                    yAxis("", Plot.axisOpts().range(min_error, max_error)).series(null, Plot.data().xy(generation, errors), null);
		try {
			String path = Repo.getLatest().toString(); 
			plot.save(path + "/line", "png");
		} catch (IOException e) {
			e.printStackTrace();
		}
    }
	
	public static void scatter(List<Double[]> data) {
		int[] arr = LinAlg.rep(0, data.size());
		List<Integer> ints = IntStream.of(arr).boxed().collect(Collectors.toList());
		scatter(data, ints, null);
    }
	
	public static void scatter(List<Double[]> data, List<Integer> target) {
		scatter(data, target, null);
	}
	
	public static void scatter(List<Double[]> data, List<Integer> target, List<Double[]> support) {
		Plot plot = Plot.plot(Plot.plotOpts());
		Map<Integer, List<Double[]>> map = zipToMap(target, data);
		
		for(Integer classId : map.keySet()) {
			List<Double[]> xy = map.get(classId);
			List<Double> xs = new ArrayList<>();
			List<Double> ys = new ArrayList<>();
			for(Double[] d : xy) {
				xs.add(d[0]); ys.add(d[1]);
			}
			DataSeriesOptions serOpt = Plot.seriesOpts().marker(Plot.Marker.CIRCLE).markerColor(colors[classId]).color(Color.BLACK).line(Line.NONE);
			plot.series(classId.toString(), Plot.data().xy(xs,ys), serOpt);
		}
		
		if(support != null) {
			List<Double> xs = new ArrayList<>();
			List<Double> ys = new ArrayList<>();
			for(Double[] point : support) {
				xs.add(point[0]);
				ys.add(point[1]);
			}
			DataSeriesOptions serOpt = Plot.seriesOpts().marker(Plot.Marker.CIRCLE).markerColor(supportColor).color(Color.BLACK).line(Line.NONE);
			plot.series("SUPPORT", Plot.data().xy(xs,ys), serOpt);
		}
		
		Path path = Repo.getLatest().resolve("scatter");
		if(Files.exists(path))
			path = Repo.getLatest().resolve("scatter_learned");
		try {
			plot.save(path.toString(), "png");
		} catch (IOException e) {
			e.printStackTrace();
		}
    }

	public static void scatter(List<Double[]> data, List<Integer> target, List<Double[]> support, int[] xrange, int[] yrange) {
		Plot plot = Plot.plot(Plot.plotOpts());
		Map<Integer, List<Double[]>> map = zipToMap(target, data);
		
		for(Integer classId : map.keySet()) {
			List<Double[]> xy = map.get(classId);
			List<Double> xs = new ArrayList<>();
			List<Double> ys = new ArrayList<>();
			for(Double[] d : xy) {
				xs.add(d[0]); ys.add(d[1]);
			}
			DataSeriesOptions serOpt = Plot.seriesOpts().marker(Plot.Marker.CIRCLE).markerColor(colors[classId]).color(Color.BLACK).line(Line.NONE);
			plot.xAxis("X", Plot.axisOpts().range(xrange[0], xrange[1])).
			     yAxis("Y", Plot.axisOpts().range(yrange[0], yrange[1])).
			     series(classId.toString(), Plot.data().xy(xs,ys), serOpt);
		}
		
		if(support != null) {
			List<Double> xs = new ArrayList<>();
			List<Double> ys = new ArrayList<>();
			for(Double[] point : support) {
				xs.add(point[0]);
				ys.add(point[1]);
			}
			DataSeriesOptions serOpt = Plot.seriesOpts().marker(Plot.Marker.CIRCLE).markerColor(supportColor).color(Color.BLACK).line(Line.NONE);
			plot.series("SUPPORT", Plot.data().xy(xs,ys), serOpt);
		}
		
		Path path = Repo.getLatest().resolve("dataset");
		try {
			plot.save(path.toString(), "png");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static <K, V> Map<K, List<V>> zipToMap(List<K> keys, List<V> values) {
		Map<K, List<V>> map = new HashMap<>();
		for(int i = 0; i < values.size(); ++i) {
			K key = keys.get(i); V label = values.get(i);
			if(map.containsKey(key)) {
				List<V> list = map.get(key);
				list.add(label);
				map.put(key, list);
			}
			else {
				List<V> list = new ArrayList<>();
				list.add(label);
				map.put(key, list);
			}
		}
	    return map;
	}

}