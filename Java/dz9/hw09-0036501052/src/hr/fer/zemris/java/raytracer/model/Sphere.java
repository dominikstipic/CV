package hr.fer.zemris.java.raytracer.model;

import static java.lang.Math.sqrt;

import hr.fer.zemris.java.raytracer.model.GraphicalObject;
import hr.fer.zemris.java.raytracer.model.Point3D;
import hr.fer.zemris.java.raytracer.model.Ray;
import hr.fer.zemris.java.raytracer.model.RayIntersection;

/**
 * Represents sphere on scene.Contains method which finds nearest inteersection with some ray.
 * @author Dominik Stipić
 *
 */
public class Sphere extends GraphicalObject{
	/**
	 * sphere center
	 */
	private Point3D center;
	/**
	 * sphere radius
	 */
	private double radius;
	/**
	 * red diffuse constant
	 */
	private double kdr;
	/**
	 * green diffuse constant
	 */
	private double kdg;
	/**
	 * blue diffuse constant
	 */
	private double kdb;
	/**
	 * red reflective constant
	 */
	private double krr;
	/**
	 * green reflective constant
	 */
	private double krg;
	/**
	 * blue reflective constant
	 */
	private double krb;
	/**
	 * describes rougness of surfaces
	 */
	private double krn;
	
	/**
	 * Creates this object
	 * @param center center of sphere
	 * @param radius radius of sphere
	 * @param kdr red diffuse constant 
	 * @param kdg green diffuse constant
	 * @param kdb blue diffuse constant
	 * @param krr red reflective constant
	 * @param krg green reflective constant
	 * @param krb blue reflective constant
	 * @param krn describes rougness of surfaces
	 */
	public Sphere(Point3D center, double radius, double kdr, double kdg, double kdb, double krr, double krg, double krb,
			double krn) {
		this.center = center;
		this.radius = radius;
		this.kdr = kdr;
		this.kdg = kdg;
		this.kdb = kdb;
		this.krr = krr;
		this.krg = krg;
		this.krb = krb;
		this.krn = krn;
	}

	@Override
	public RayIntersection findClosestRayIntersection(Ray ray) {
		Point3D S = ray.start;
		Point3D D = ray.direction;
		
		double A = 1;
		double B = 2*(D.x*(S.x - center.x) + D.y*(S.y - center.y) + D.z*(S.z - center.z)) ;
		double C = (S.x - center.x)*(S.x - center.x) + (S.y - center.y)*(S.y - center.y) + (S.z - center.z)*(S.z - center.z) - radius*radius;
		
		double determinante = B*B - 4*A*C;
		if(determinante < 0) return null;
		
		Double t0 = (double)(-B + sqrt(determinante))/(2*A);
		Double t1 = (double)(-B - sqrt(determinante))/(2*A);
		 //sjecista zraka-sfera
		Point3D first = new Point3D(S.x+t0*D.x,S.y+t0*D.y,S.z+t0*D.z);
		Point3D second = new Point3D(S.x+t1*D.x,S.y+t1*D.y,S.z+t1*D.z);
		Point3D start = ray.start;
		
		if(first.sub(start).norm() <= second.sub(start).norm()) {
			return new SphereIntersection(first, first.sub(start).norm(), true); 
		}
		return new SphereIntersection(second, second.sub(start).norm(), true);
	}
	
	/**
	 * Models intersection of sphere with some ray
	 * @author Dominik Stipić
	 *
	 */
	private class SphereIntersection extends RayIntersection{
		/**
		 * Creates intersection
		 * @param point of intersection
		 * @param distance distance from ray start
		 * @param outer is outer intersection
		 */
		public SphereIntersection(Point3D point, double distance, boolean outer) {
			super(point, distance, outer);
		}
		
		@Override
		public Point3D getNormal() {
			return super.getPoint().sub(center).normalize();
		}

		@Override
		public double getKdr() {
			return kdr;
		}

		@Override
		public double getKdg() {
			return kdg;
		}

		@Override
		public double getKdb() {
			return kdb;
		}

		@Override
		public double getKrr() {
			return krr;
		}

		@Override
		public double getKrg() {
			return krg;
		}

		@Override
		public double getKrb() {
			return krb;
		}

		@Override
		public double getKrn() {
			return krn;
		}
		
	}
	

	
}
