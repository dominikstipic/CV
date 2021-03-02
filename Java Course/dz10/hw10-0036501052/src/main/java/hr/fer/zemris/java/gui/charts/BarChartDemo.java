package hr.fer.zemris.java.gui.charts;

import java.awt.BorderLayout;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingUtilities;

/**
 * Demonstrates usage of {@link BartChartComponent}.
 * All needed information is provided through command line iterface as path to text file.
 * @author Dominik Stipić
 *
 */
public class BarChartDemo extends JFrame{
	private static final long serialVersionUID = 1L;
	/**
	 * width of frame
	 */
	private static final int X = 600;
	/**
	 * height of frame
	 */
	private static final int Y = 600;
	/**
	 * chart
	 */
	private BarChart chart;

	/**
	 * Constructs this demo
	 * @param chart with all info
	 * @param p path to file 
	 */
	public BarChartDemo(BarChart chart, Path p) {
		this.chart = Objects.requireNonNull(chart,"BarChart cannot be null");
		setTitle("chart");
		setSize(X, Y);
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		getContentPane().add(new JLabel(p.toString()),BorderLayout.NORTH);
		initGui();
	}
	
	/**
	 * init chart gui
	 */
	public void initGui() {
		BartChartComponent c = new BartChartComponent(chart);
		getContentPane().add(c);
	}
	
	/**
	 * Automatically starts,expects arguments from command line interface
	 * @param args path to file
	 */
	public static void main(String[] args) {
		if(args.length != 1) {
			System.out.println("Argument not provided");
			System.exit(1);
		}
		Path p = Paths.get(args[0]);
		if(!Files.isRegularFile(p)) {
			System.out.println("Regular file not provided");
			System.exit(1);
		}

		try {
			BarChart chart = getBarChart(p);
			SwingUtilities.invokeLater(()->{
				new BarChartDemo(chart,p).setVisible(true);
			});
		} catch (Exception e) {
			System.out.println("error while processing charts");
			System.exit(1);
		}
	}
	
	public static BarChart getBarChart(Path p) throws IOException {
		BarChart chart;
		try(BufferedReader reader = Files.newBufferedReader(p)){
			String xText = reader.readLine();
			String yText = reader.readLine();
			String parts[] = reader.readLine().split("[ ]+");
			List<XYValue> list = new LinkedList<>();
			for(String str:parts) {
				String [] tokens = str.split(",");
				int x = Integer.valueOf(tokens[0]);
				int y = Integer.valueOf(tokens[1]);
				list.add(new XYValue(x, y));
			}
				
			int minY = Integer.valueOf(reader.readLine());
			int maxY = Integer.valueOf(reader.readLine());
			int delta = Integer.valueOf(reader.readLine());
			
			chart = new BarChart(list, xText, yText, minY, maxY, delta);
		}
		return chart;
	}
	
}
