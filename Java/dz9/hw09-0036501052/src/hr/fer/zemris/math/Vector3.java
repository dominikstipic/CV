package hr.fer.zemris.math;
import static java.lang.Math.sqrt;

import java.util.Objects;

/**
 * Models Vector with 3 components and porvides typically methods for vector handling
 * @author Dominik Stipić
 *
 */
public class Vector3 {
	/**
	 * x component
	 */
	private final double x;
	/**
	 * y component
	 */
	private final double y;
	/**
	 * z component
	 */
	private final double z;
	
	/**
	 * Creates Vector
	 * @param x component
	 * @param y component
	 * @param z component
	 */
	public Vector3(double x, double y, double z) {
		this.x = x;
		this.y = y;
		this.z = z;
	} 
	
	/**
	 * Calculates vector length
	 * @return length
	 */
	public double norm() {
		return sqrt(x*x + y*y + z*z);
	} 
	
	/**
	 * retruns normalized vector
	 * @return new normalized vector
	 */
	public Vector3 normalized() {
		double abs = norm();
		return new Vector3((double)x/abs, (double)y/abs, (double)z/abs);
	} 
	
	/**
	 * Adds two vectors
	 * @param other other vector
	 * @return result of adding
	 */
	public Vector3 add(Vector3 other) {
		Objects.requireNonNull(other, "Vector cannot be null");
		return new Vector3(x+other.x, y+other.y, z+other.z);
	} 
	
	/**
	 * Subtracts two vectors
	 * @param other vector
	 * @return result of subtraction
	 */
	public Vector3 sub(Vector3 other) {
		Objects.requireNonNull(other, "Vector cannot be null");
		return new Vector3(x-other.x, y-other.y, z-other.z);
	} 
	
	/**
	 * performes dot product with 2 vectors
	 * @param other vector
	 * @return scalar 
	 */
	public double dot(Vector3 other) {
		Objects.requireNonNull(other, "Vector cannot be null");
		return x*other.x + y*other.y + z*other.z;
	}
	
	/**
	 * performes cross product with 2 vectors
	 * @param other vector
	 * @return vector
	 */
	public Vector3 cross(Vector3 other) {
		Objects.requireNonNull(other, "Vector cannot be null");
		double newX = y*other.z - z*other.y;
		double newY = z*other.x - x*other.z;
		double newZ = x*other.y - y*other.x;
		return new Vector3(newX, newY, newZ);
	}
	
	/**
	 * Scales given vector by some factor
	 * @param s factor
	 * @return scaled vector
	 */
	public Vector3 scale(double s) {
		return new Vector3(s*x, s*y, s*z);
	} 
	
	/**
	 * calculates cosine of two vectors 
	 * @param other
	 * @return cosine of angle beetwen vectors
	 */
	public double cosAngle(Vector3 other) {
		Objects.requireNonNull(other, "Vector cannot be null");
		return (double) dot(other)/(norm()*other.norm());
	} 
	
	/**
	 * @return returns x component
	 */
	public double getX() {
		return x;
	}
	
	/**
	 * @return returns y component
	 */
	public double getY() {
		return y;
	}
	
	/**
	 * @return returns z component
	 */
	public double getZ() {
		return z;
	}
	
	/**
	 * returns array of vector components
	 * @return arrray of components
	 */
	public double[] toArray() {
		return new double[] {x,y,z};
	}

	@Override
	public String toString() {
		return String.format("(%6f, %.6f, %.6f)", x, y, z);
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		long temp;
		temp = Double.doubleToLongBits(x);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(y);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(z);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Vector3 other = (Vector3) obj;
		if (Double.doubleToLongBits(x) != Double.doubleToLongBits(other.x))
			return false;
		if (Double.doubleToLongBits(y) != Double.doubleToLongBits(other.y))
			return false;
		if (Double.doubleToLongBits(z) != Double.doubleToLongBits(other.z))
			return false;
		return true;
	}

	
	
}
