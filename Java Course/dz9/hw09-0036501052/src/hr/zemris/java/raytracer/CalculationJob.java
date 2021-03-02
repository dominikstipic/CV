package hr.zemris.java.raytracer;

import static java.lang.Math.pow;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveAction;

import hr.fer.zemris.java.raytracer.model.GraphicalObject;
import hr.fer.zemris.java.raytracer.model.LightSource;
import hr.fer.zemris.java.raytracer.model.Point3D;
import hr.fer.zemris.java.raytracer.model.Ray;
import hr.fer.zemris.java.raytracer.model.RayIntersection;
import hr.fer.zemris.java.raytracer.model.Scene;

/**
 * Models job which thread performes in ray-tracing algorithm
 * @author Dominik Stipić
 *
 */
public class CalculationJob extends RecursiveAction{
	private static final long serialVersionUID = 1L;
	/**
	 * red color
	 */
	private short[] red;
	/**
	 * green color
	 */
	private short[] green; 
	/**
	 * blue color
	 */
	private short[] blue;
	/**
	 * thread starting y position
	 */
	private int yMin = 0;
	/**
	 * thread last y position
	 */
	private int yMax ;
	/**
	 * Coordinate system xAxis
	 */
	private Point3D xAxis;
	/**
	 * Coordinate system yAxis
	 */
	private Point3D yAxis;
	/**
	 * Coordinate system zAxis
	 */
	private Point3D zAxis;
	/**
	 * eye position
	 */
	private Point3D eye;
	/**
	 * horizonatl screen length
	 */
	private double horizontal;
	/**
	 * vertical screen length
	 */
	private double vertical;
	/**
	 * width of screene
	 */
	private int width;
	/**
	 * height of screen
	 */
	private int height;
	/**
	 * scene
	 */
	private Scene s;
	/**
	 * corner of screen
	 */
	private Point3D corner;
	/**
	 * number of y lines that one thread processes
	 */
	static final int treshold = 16;
	
	/**
	 * Creates object
	 * @param red color
	 * @param green color
	 * @param blue color
	 * @param yMin thread min y
	 * @param yMax thread max y
	 * @param horizontal length 
	 * @param vertical length
	 * @param x axis 
	 * @param y axis 
	 * @param z axis
	 * @param eye posiiton
	 * @param width of screeen
	 * @param height of screen 
	 * @param s scene
	 * @param corner corner of coordinate system
	 */
	public CalculationJob(short[] red, short[] green, short[] blue, int yMin, int yMax, 
			              double horizontal, double vertical, Point3D x, Point3D y, Point3D z,Point3D eye, int width,int height,Scene s,Point3D corner) {
		this.yMin = yMin;
		this.yMax = yMax;
		this.red = red;
		this.green = green;
		this.blue = blue;
		this.horizontal = horizontal;
		this.vertical = vertical;
		xAxis = x;
		yAxis = y;
		zAxis = z;
		this.eye = eye;
		this.width = width;
		this.height = height;
		this.s = s;
		this.corner = corner;
	}
	
	@Override
	protected void compute() {
		if(yMax-yMin <= treshold) {
			computeDirect();
			return;
		}
		invokeAll(
				new CalculationJob(red, green, blue, yMin, yMin+(yMax-yMin)/2, horizontal, vertical, xAxis, yAxis, zAxis, eye, width, height, s, corner),
				new CalculationJob(red, green, blue, yMin+(yMax-yMin)/2, yMax, horizontal, vertical, xAxis, yAxis, zAxis, eye, width, height, s, corner)
			);
	}

	/**
	 * Finds cloosest intersection with eye ray and colors it
	 * @param s scene
	 * @param ray eye-screen view
	 * @param rgb value
	 */
	public static void tracer(Scene s, Ray ray, short[] rgb) {
		List<GraphicalObject> objects = s.getObjects();
		for(GraphicalObject o : objects) {
			//sjeciste objekt-pixel
			RayIntersection intersection = o.findClosestRayIntersection(ray);
			if(intersection == null) {
				rgb[0] = 0;
				rgb[1] = 0;
				rgb[2] = 0;
				continue;
			}
			List<GraphicalObject> otherObjects = new ArrayList<>(objects);
			otherObjects.remove(o);
			determineColorFor(intersection, otherObjects, s, rgb, ray);
			return;
		}
		
	}
	
	/**
	 * renders image
	 */
	private void computeDirect() {
		short[] rgb = new short[3];
		int offset = yMin * width;
		for (int y = yMin; y < yMax; y++) {
			for (int x = 0; x < width; x++) {
				double forX = (double)x*horizontal/(width-1);
				double forY = (double)y*vertical/(height-1);
				Point3D screenPoint = corner.add(xAxis.scalarMultiply(forX)).sub(yAxis.scalarMultiply(forY));
				Ray ray = Ray.fromPoints(eye, screenPoint);

				tracer(s, ray, rgb);
				
				red[offset] = rgb[0] > 255 ? 255 : rgb[0];
				green[offset] = rgb[1] > 255 ? 255 : rgb[1];
				blue[offset] = rgb[2] > 255 ? 255 : rgb[2];
				offset++;
			}
		}
	}
	
	/**
	 * Determines color for given intersection of eye ray
	 * @param intersection with some object
	 * @param objects other objects in scene 
	 * @param s scene
	 * @param rgb red-green-blue value
	 * @param eyeRay ray from eyes
	 */
	public static void determineColorFor(RayIntersection intersection, List<GraphicalObject> objects ,Scene s, short rgb[], Ray eyeRay) {
		rgb[0] = 15;
		rgb[1] = 15;
		rgb[2] = 15;
		
		for(LightSource light : s.getLights()) {
			Ray r = Ray.fromPoints(light.getPoint(), intersection.getPoint());
			boolean obstacle = false;
			for(GraphicalObject o : objects) {
				RayIntersection otherIntersection = o.findClosestRayIntersection(r); 
				if(otherIntersection == null)continue;
				if(otherIntersection.getDistance() <= intersection.getDistance()) {
					obstacle = true;
					break;
				}
			}
			if(obstacle)continue;
			diffuseComponent(light,r ,intersection, rgb);
			reflectiveComponent(light,r ,eyeRay, intersection, rgb);
		}
	}
	
	/**
	 * Calculates diffuse component
	 * @param source light source
	 * @param sourceRay source ray
	 * @param intersection intersectionof source ray with object
	 * @param rgb red-greeen-blue value
	 */
	public static void diffuseComponent(LightSource source, Ray sourceRay, RayIntersection intersection, short [] rgb) {
		double cosTheta = sourceRay.direction.scalarProduct(intersection.getNormal()); 
		cosTheta = cosTheta > 0 ? cosTheta : 0;
		rgb[0] += source.getR() * intersection.getKdr() * cosTheta;
		rgb[1] += source.getG() * intersection.getKdg() * cosTheta;
		rgb[2] += source.getB() * intersection.getKdb() * cosTheta;
	}
	
	/**
	 * Calculates reflection component
	 * @param source light source
	 * @param sourceRay source ray
	 * @param eyeRay ray from eye
	 * @param intersection intersection of source ray with object
	 * @param rgb red-greeen-blue value
	 */
	public static void reflectiveComponent(LightSource source,Ray sourceRay ,Ray eyeRay, RayIntersection intersection, short [] rgb) {
		double scalar = sourceRay.direction.scalarProduct(intersection.getNormal());
		Point3D reflect = sourceRay.direction.normalize().sub(intersection.getNormal().normalize().scalarMultiply(2*scalar));
		double cosAlpha = reflect.normalize().scalarProduct(eyeRay.direction.normalize());
		cosAlpha = cosAlpha > 0 ? cosAlpha : 0;
		double x = pow(cosAlpha, intersection.getKrn());
		rgb[0] += source.getR() * intersection.getKrr() * x;
		rgb[1] += source.getG() * intersection.getKrg() * x;
		rgb[2] += source.getB() * intersection.getKrb() * x;
		
	}
}
