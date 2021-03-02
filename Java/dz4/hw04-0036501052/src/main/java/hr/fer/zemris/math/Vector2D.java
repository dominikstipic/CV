package hr.fer.zemris.math;

import java.util.Objects;
import static java.lang.Math.atan;
import static java.lang.Math.cos;
import static java.lang.Math.sin;
import static java.lang.Math.toDegrees;
import static java.lang.Math.toRadians;
import static java.lang.Math.abs;
import static java.lang.Math.sqrt;
import static java.lang.Math.round;

/**
 * Class which represents vector and typical matehematical operations needed for manipulating 
 * with vector.Mathematical description of state of thsi vector are : Cartesian coordiantes and positive-angle 
 * which this vector closes with x-axis. Angle is from interval [0-359] 
 * @author Dominik Stipic
 *
 */
public class Vector2D {
	/**
	 * x coordinate of this vector
	 */
	private double x;
	/**
	 * y coordinate of this vector
	 */
	private double y;
	/**
	 * positive angle beetwen [0-359]
	 */
	private double theta;
	/**
	 * Maximal allowed angle
	 */
	private static final int MAX_ANGLE = 360;
	/**
	 * Field used for double comparison
	 */
	public static final double ZERO_INTERVAL = 10E-10;

	/**
	 * Creates vector from Cartesian coordinates
	 * @param x coordinate
	 * @param y coordinate
	 */
	public Vector2D(double x, double y) {
		this.x = x;
		this.y = y;
		roundUp(); // needs updated x,y for calculateAngle
		theta = calculateAngle(x, y);
		roundUp(); // update theta

	}

	/**
	 * Getter for anlge of vector
	 * @return angle in degrees
	 */
	public double getAngle() {
		return theta;
	}
	
	/**
	 * Getter for X coordinate of vector
	 * @return x coordinate
	 */
	public double getX() {
		return x;
	}

	/**
	 * Getter for Y coordinate of vector
	 * @return y coordinate
	 */
	public double getY() {
		return y;
	}

	/**
	 * Static builder method used for building vector from its polar representation
	 * @param magnitude of vector
	 * @param angle of vector in degrees
	 * @return newly created vector
	 */
	public static Vector2D fromMagnitudeAndAngle(double magnitude, double angle) {
		double x = magnitude * cos(toRadians(angle));
		double y = magnitude * sin(toRadians(angle));
		return new Vector2D(x, y);
	}
	
	/**
	 * Method for normalizing this vector.
	 * Normalization is procces of adjusting coordinates 
	 * of vector so that his magnitude becomes one.Direction stays the same  
	 * as initial vector.  
	 */
	public void normalize() {
		double magnitude = sqrt(x * x + y * y);
		if (magnitude <= ZERO_INTERVAL) {
			throw new ArithmeticException("cannot normalize point vector!");
		}
		x /= magnitude;
		y /= magnitude;

		roundUp();
	}

	/**
	 * Tranlates this vector to new origin
	 * @param offset Vector
	 */
	public void translate(Vector2D offset) {
		Objects.requireNonNull(offset);
		x += offset.x;
		y += offset.y;

		roundUp();
	}

	/**
	 * Returns translated angle to new origin
	 * @param offset Vector
	 * @return translated vector
	 */
	public Vector2D translated(Vector2D offset) {
		Objects.requireNonNull(offset);
		return new Vector2D(x + offset.x, y + offset.y);
	}

	/**
	 * Returns rotated vector for given angle 
	 * @param angle - needed for rotation 
	 */
	public void rotate(double angle) {
		Vector2D newVec = rotated(angle);
		x = newVec.x;
		y = newVec.y;
		theta = newVec.theta;

		roundUp();
	}

	/**
	 * Returns rotated vector for given angle withaout changing this vector
	 * @param angle - needed for rotation 
	 * @return rotated vector
	 */
	public Vector2D rotated(double angle) {
		double rad = toRadians(angle % MAX_ANGLE);
		double newX;
		double newY;
		if (rad > 0) {
			newX = x * cos(rad) - y * sin(rad);
			newY = x * sin(rad) + y * cos(rad);
		} else {
			rad *= -1;
			newX = x * cos(rad) + y * sin(rad);
			newY = -x * sin(rad) + y * cos(rad);
		}

		return new Vector2D(checkIfZero(newX), checkIfZero(newY));
	}

	/**
	 * Scales this vector for given scaler
	 * @param scaler
	 */
	public void scale(double scaler) {
		x *= scaler;
		y *= scaler;

		roundUp();
	}

	/**
	 * It scales this vector by given scaler withaout changing state of this vector.
	 * @param scaler stretching factor
	 * @return scaled vector
	 */
	public Vector2D scaled(double scaler) {
		return new Vector2D(x * scaler, y * scaler);
	}

	/**
	 * @return copy of this vector
	 */
	public Vector2D copy() {
		return new Vector2D(x, y);
	}

	/**
	 * returns angle in representaion which is from 0 to 359
	 * It also checks boundary conditions
	 * @param x coordindate 
	 * @param y coordinate
	 * @return correspodent angle of given coordinates
	 */
	private static double calculateAngle(double x, double y) {
		double theta = toDegrees(atan(y / x));
		if (x == -1 && y == 0) {
			return theta = MAX_ANGLE / 2;
		} else if (abs(x) <= ZERO_INTERVAL && y > 0) {
			return theta = MAX_ANGLE / 4;
		} else if (x == 0 && y < 0) {
			return theta = ((double) 3 / 4) * MAX_ANGLE;
		} else if (x == 0 && y == 0) {
			return theta = 0;
		} else if (x < 0 && y >= 0) { // 2. quadrant
			return theta += MAX_ANGLE / 2;
		} else if (x < 0 && y < 0) { // 3. quadrant
			return theta += MAX_ANGLE / 2;
		} else if (x > 0 && y < 0) { // 4. quadrant
			return theta += MAX_ANGLE;
		} else {
			return theta;
		}

	}

	/**
	 * Adds modularly angles and keeping the result angle in defined interval which is
	 * [0-359]
	 * @param angle 
	 * @param delta
	 * @return result angle from specified interval
	 */
	public static double addModularly(double angle, double delta) {
		delta %= MAX_ANGLE;
		if (delta < 0) {
			delta = MAX_ANGLE + delta;
		}
		return (delta + angle) % MAX_ANGLE;
	}

	private double checkIfZero(double value) {
		if (abs(value) < ZERO_INTERVAL) {
			return 0;
		}
		return value;
	}

	/**
	 * Rounds up double who are nearly integers. Treshold for rounding up is 
	 * when double is smaller or bigger by specified ZERO_INTERVAL
	 */
	public void roundUp() {
		if (abs(abs(round(theta)) - abs(theta)) <= ZERO_INTERVAL) {
			theta = round(theta);
		}
		if (abs(abs(round(x)) - abs(x)) <= ZERO_INTERVAL) {
			x = round(x);
		}
		if (abs(abs(round(y)) - abs(y)) <= ZERO_INTERVAL) {
			y = round(y);
		}
	}

	@Override
	public String toString() {
		return x + "i" + " + " + y + "j";
	}

	@Override
	public boolean equals(Object obj) {
		if (!(obj instanceof Vector2D))
			return false;
		Vector2D other = (Vector2D) obj;
		if (abs(this.x - other.x) > ZERO_INTERVAL) {
			return false;
		}
		if (abs(this.y - other.y) > ZERO_INTERVAL) {
			return false;
		}
		return true;
	}

	

}
