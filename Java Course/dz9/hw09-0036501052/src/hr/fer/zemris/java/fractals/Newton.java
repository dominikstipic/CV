package hr.fer.zemris.java.fractals;

import static java.lang.Math.abs;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.Scanner;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import hr.fer.zemris.java.fractals.viewer.FractalViewer;
import hr.fer.zemris.java.fractals.viewer.IFractalProducer;
import hr.fer.zemris.java.fractals.viewer.IFractalResultObserver;
import hr.fer.zemris.math.Complex;
import hr.fer.zemris.math.ComplexRootedPolynomial;

/**
 * Calculates fracatal based on the Complex numbers which user inputs.
 * Calculation is preformed using Newton-Raphson Method
 * @author Dominik Stipić
 *
 */
public class Newton implements IFractalProducer{
	
	/**
	 * Rooted polynomial
	 */
	ComplexRootedPolynomial roots;
	
	/**
	 * Maximal number of newton-raphson algorithm iteration
	 */
	private static final int MAX_ITER = 16 * 16 * 16;
	
	/**
	 * distance to root
	 */
	private static final double MODULE_LIMIT = 1E-3;
	
	/**
	 * convergence limit
	 */
	private static final double CLOSENESS_LIMIT = 1E-3;
	
	/**
	 * number of thread y tracks
	 */
	private static final int TRACK_NUM = 8 * Runtime.getRuntime().availableProcessors();
	
	/**
	 * image data
	 */
	private short[] data;
	
	
	/**
	 * Creates Newton Object
	 * @param roots polynomial accoreding to image will be drawn
	 */
	public Newton(ComplexRootedPolynomial roots) {
		Objects.requireNonNull(roots, "Polynome cannot be null");
		this.roots = roots;
	}

	@Override
	public void produce(double reMin, double reMax, double imMin, double imMax, int width, int height, long requestNo,
			IFractalResultObserver observer) {
		
		data = new short[width * height];
		int numberOfThreads = height / TRACK_NUM;
		int threadBorder = height / numberOfThreads;
		
		ExecutorService executor = Executors.newFixedThreadPool(numberOfThreads);
		List <Future<Void>> fut = new ArrayList<>();
		
		for(int i = 0; i < numberOfThreads; ++i) {
			int yMin = i*threadBorder;
			int yMax = (i+1)*threadBorder - 1;
			if(i == numberOfThreads - 1) {
				yMax = height - 1;
			}
				CalculationJob job = new CalculationJob(reMin, reMax, imMin, imMax, width, height,yMin,yMax,data);
				fut.add(executor.submit(job));
			}
		
			for(Future<Void> f : fut) {
				try {
					f.get();
				} catch (InterruptedException | ExecutionException e) {
					throw new RuntimeException(e);
				}
			}
			executor.shutdown();
			observer.acceptResult(data, (short)(roots.toComplexPolynom().order() + 1), requestNo);
		}


	
	/**
	 * Thread job.Thread executes job of calculating data color with Newthon-Raphson algorithm.
	 * Each thread gets the portion of y range on which it calculates.
	 * @author Dominik Stipić
	 *
	 */
	private class CalculationJob implements Callable<Void>{
		
		/**
		 * min complex real part
		 */
		double reMin;
		/**
		 * max complex real part
		 */
		double reMax;
		/**
		 * min complex imag part
		 */
		double imMin;
		/**
		 * max complex imag part
		 */
		double imMax;
		/**
		 * width of screen
		 */
		int width;
		/**
		 * height of screen
		 */
		int height;
		/**
		 * thread ymin
		 */
		int yMin;
		/**
		 * thread yMax
		 */
		int yMax;
		/**
		 * data
		 */
		short[] data;

		/**
		 * Creates Calculatin Job
		 * @param reMin
		 * @param reMax
		 * @param imMin
		 * @param imMax
		 * @param width
		 * @param height
		 * @param yMin
		 * @param yMax
		 * @param data
		 */
		public CalculationJob(double reMin, double reMax, double imMin, double imMax, int width, int height, int yMin,
				int yMax, short[] data) {
			this.reMin = reMin;
			this.reMax = reMax;
			this.imMin = imMin;
			this.imMax = imMax;
			this.width = width;
			this.height = height;
			this.yMin = yMin;
			this.yMax = yMax;
			this.data = data;
		}



		@Override
		public Void call() throws Exception {
			int offset = yMin * width;
			
			for(int y = yMin; y <= yMax; y++) {
				for(int x = 0; x < width; x++) {
					Complex c = new Complex(x / (width-1.0) * (reMax - reMin) + reMin, (height-1.0-y) / (height-1) * (imMax - imMin) + imMin);
					Complex zn = c;
					int iter = 0;
					double module;
					
					do {
						Complex fraction = roots.apply(zn).divide(roots.toComplexPolynom().derive().apply(zn));
						Complex zn1 = zn.sub(fraction);
						module = zn.sub(zn1).module(); 
						zn = zn1;
						++iter;
					} while(abs(module) > CLOSENESS_LIMIT && iter < MAX_ITER);
					int index = roots.indexOfClosestRootFor(zn, MODULE_LIMIT);
					if(index == -1) {
						data[offset++] = 0;
					}
					else {
						data[offset++] = (short)(index+1);
					}
				}
			}
			return null;
		}
		
	}
	
	/**
	 * method which automatically starts 
	 * @param args -
	 */
	public static void main(String[] args) {
		System.out.println("Welcome to Newton-Raphson iteration-based fractal viewer.");
		System.out.println("Please enter at least two roots, one root per line. Enter 'done' when done.");
		
		ComplexRootedPolynomial roots = readInput();
		System.out.println("Image of fractal will appear shortly.Thank you");
		
		FractalViewer.show(new Newton(roots));
	}

	/**
	 * reads user input
	 * @return Complex rootted polynomial
	 */
	public static ComplexRootedPolynomial readInput(){
		List<Complex> list = new LinkedList<>();
		try(Scanner s = new Scanner(System.in)){
			int count = 0;
			while(true) {
				String str = s.nextLine().trim();
				if(str.equals("done")) {
					if(count < 2) {
						System.out.println("insufficient number of inputs");
						continue;
					}
					break;
				} 
				try {
					list.add(parseComplex(str));
					++count;
				} catch (Exception e) {
					System.out.println("Invalid input,try again");
				}
			}
			
		}
		return new ComplexRootedPolynomial(list.toArray(new Complex[0]));
	}
	
	/** 
	 * Parses user input into complex number
	 * @param line which user inputed
	 * @return parsed Complex number
	 */
	public static Complex parseComplex(String line) {
		line = line.replaceAll("[ ]+", "");
		if(!line.contains("i")) {
			Double re = Double.parseDouble(line);
			return new Complex(re, 0);
		}
		if(line.matches("(-)?i")) {
			if(line.contains("-")) {
				return Complex.IM_NEG;
			}
			return Complex.IM;
		}
		if(line.matches("((-)?\\d+(\\d+)?)?i")){
			Double im = Double.parseDouble(line.substring(0,line.indexOf("i")));
			return new Complex(0 ,im);
		}
		String re = "";
		if(line.startsWith("-")) {
			re = "-"; 
			line = line.substring(1);
		}
		String sign = "+";
		if(line.contains("-"))sign = "-";
		re += line.substring(0, line.indexOf(sign));
		String im = line.substring(line.indexOf(sign)+1);
		im = im.replaceAll("i", "");
		if(im.isEmpty())im = "1";
		
		return new Complex(Double.parseDouble(re), Double.parseDouble(im)); 
	}
	
	
}
