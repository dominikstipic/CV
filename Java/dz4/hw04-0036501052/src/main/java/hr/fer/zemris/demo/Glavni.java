package hr.fer.zemris.demo;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;

import hr.fer.zemris.java.custom.collections.ArrayIndexedCollection;
import hr.fer.zemris.lsystem.impl.LSystemBuilderImpl;
import hr.fer.zemris.lsystems.LSystem;
import hr.fer.zemris.lsystems.LSystemBuilderProvider;
import hr.fer.zemris.lsystems.gui.LSystemViewer;

/**
 * Demo for representation of LSystem fractals 
 * Scripts for genereting fractals are situted in "src/main/resources" 
 * @author Dominik Stipic
 *
 */
public class Glavni {
	static final String HILBERT_CURVE = "src/main/resources/hilbertCurve.txt";
	static final String KOCH2 = "src/main/resources/koch2.txt"; 
	static final String KOCH_CURVE = "src/main/resources/kochCurve.txt";
	static final String KOCH_ISLAND = "src/main/resources/kochIsland.txt";
	static final String PLANT1 = "src/main/resources/plant1.txt";
	static final String PLANT2 = "src/main/resources/plant2.txt";
	static final String PLANT3 = "src/main/resources/plant3.txt";
	static final String SIERPINSKI_GASKET = "src/main/resources/sierpinskiGasket.txt";
	
	/**
	 * Method which is automaticaly launched when program starts 
	 */
	public static void main(String[] args) {
		//LSystemViewer.showLSystem(createKochCurve(LSystemBuilderImpl::new));
		//LSystemViewer.showLSystem(hilbertCurve(LSystemBuilderImpl::new));
		LSystemViewer.showLSystem(readFractal(LSystemBuilderImpl::new, PLANT1));
	}
	
	/**
	 * Reads script from given path and returns class which creates certain fractal
	 * @param provider which builds configuration
	 * @param path to scrpits
	 * @return LSystem configuration
	 */
	private static LSystem readFractal(LSystemBuilderProvider provider, String path) {
		try {
			Scanner s = new Scanner(new File(path));
			ArrayIndexedCollection col = new ArrayIndexedCollection();
			while(true) {
				String line;
				try {
					line = s.nextLine();
					if(line == null)break;
					col.add(line);
				} catch (Exception e) {
					break;
				}
				
			}
			Object[] obj = col.toArray();
			String []data = Arrays.copyOf(obj, obj.length, String[].class);
			s.close();
			
			return provider.createLSystemBuilder().configureFromText(data).build();
		} catch (FileNotFoundException e) {
			System.out.println("file not found");
			System.exit(1);
		}
		return null;
		
	}
	
	//demos scripts and configurations listed below
	
	private static LSystem createKochCurve(LSystemBuilderProvider provider) {
		return provider.createLSystemBuilder()
				.registerCommand('F', "draw 1")
				.registerCommand('+', "rotate 60")
				.registerCommand('-', "rotate -60")
				.setOrigin(0.05, 0.4)
				.setAngle(0)
				.setUnitLength(0.9)
				.setUnitLengthDegreeScaler(1.0/3.0)
				.registerProduction('F', "F+F--F+F")
				.setAxiom("F")
				.build();
		}
	
	private static LSystem hilbertCurve(LSystemBuilderProvider provider) {
		return provider.createLSystemBuilder()
				.registerCommand('F', "draw 1")
				.registerCommand('+', "rotate 90")
				.registerCommand('-', "rotate -90")
				.setOrigin(0.05, 0.05)
				.setAngle(0)
				.setUnitLength(0.9)
				.setUnitLengthDegreeScaler(1.0/2.0)
				.registerProduction('L', "+RF-LFL-FR+")
				.registerProduction('R', "-LF+RFR+FL-")
				.setAxiom("L")
				.build();
	}
	
	
	
}
