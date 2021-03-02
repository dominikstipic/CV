package hr.fer.zemris.java.raytracer.model;

import java.util.ArrayList;
import java.util.List;
import hr.fer.zemris.java.raytracer.model.GraphicalObject;
import hr.fer.zemris.java.raytracer.model.IRayTracerProducer;
import hr.fer.zemris.java.raytracer.model.IRayTracerResultObserver;
import hr.fer.zemris.java.raytracer.model.LightSource;
import hr.fer.zemris.java.raytracer.model.Point3D;
import hr.fer.zemris.java.raytracer.model.Ray;
import hr.fer.zemris.java.raytracer.model.RayIntersection;
import hr.fer.zemris.java.raytracer.model.Scene;
import hr.fer.zemris.java.raytracer.viewer.RayTracerViewer;
import static java.lang.Math.pow;

/**
 * Represents Ray-casting algorithm which is used for image rendering.
 * This example renders sphere and colors it with Phong model.
 * @author Dominik Stipić
 *
 */
public class RayCaster {
	
	/**
	 * autommaticaly starts
	 * @param args
	 */
	public static void main(String[] args) {
		RayTracerViewer.show(getIRayTracerProducer(), new Point3D(10, 0, 0), new Point3D(0, 0, 0),
				new Point3D(0, 0, 10), 20, 20);
	}

	/**
	 * Produces object which knows to draw itself
	 * @return IRayTracerProducer
	 */
	private static IRayTracerProducer getIRayTracerProducer() {
		return new IRayTracerProducer() {
			@Override
			public void produce(Point3D eye, Point3D view, Point3D viewUp, double horizontal, double vertical,
					int width, int height, long requestNo, IRayTracerResultObserver observer) {
				System.out.println("Započinjem izračune...");
				short[] red = new short[width * height];
				short[] green = new short[width * height];
				short[] blue = new short[width * height];

				Point3D zAxis = view.sub(eye).normalize();
				Point3D yAxis = viewUp.normalize().sub(zAxis.scalarMultiply(zAxis.scalarProduct(viewUp.normalize()))).normalize();
				Point3D xAxis = zAxis.vectorProduct(yAxis).normalize();
				Point3D screenCorner = view.sub(xAxis.scalarMultiply(horizontal/2)).add(yAxis.scalarMultiply(vertical/2));
				
				Scene scene = RayTracerViewer.createPredefinedScene();
				short[] rgb = new short[3];
				int offset = 0;
				for (int y = 0; y < height; y++) {
					for (int x = 0; x < width; x++) {
						double forX = (double)x*horizontal/(width-1);
						double forY = (double)y*vertical/(height-1);
						Point3D screenPoint = screenCorner.add(xAxis.scalarMultiply(forX)).sub(yAxis.scalarMultiply(forY));
						Ray ray = Ray.fromPoints(eye, screenPoint);

						tracer(scene, ray, rgb);
						
						red[offset] = rgb[0] > 255 ? 255 : rgb[0];
						green[offset] = rgb[1] > 255 ? 255 : rgb[1];
						blue[offset] = rgb[2] > 255 ? 255 : rgb[2];
						offset++;
					}
				}
				System.out.println("Izračuni gotovi...");
				observer.acceptResult(red, green, blue, requestNo);
				System.out.println("Dojava gotova...");
			}
		};
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
		if(cosTheta < 0 ) cosTheta = 0;
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
		if(cosAlpha < 0 ) cosAlpha = 0;
		cosAlpha = cosAlpha > 0 ? cosAlpha : 0;
		double x = pow(cosAlpha, intersection.getKrn());
		rgb[0] += source.getR() * intersection.getKrr() * x;
		rgb[1] += source.getG() * intersection.getKrg() * x;
		rgb[2] += source.getB() * intersection.getKrb() * x;
		
	}
	
}
