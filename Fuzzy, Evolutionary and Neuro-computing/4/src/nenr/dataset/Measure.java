package nenr.dataset;

public class Measure {
	private double x,y,f;
	
	public Measure(double x, double y, double f) {
		this.x = x;
		this.y = y;
		this.f = f;
	}

	public double getX() {
		return x;
	}



	public void setX(double x) {
		this.x = x;
	}



	public double getY() {
		return y;
	}



	public void setY(double y) {
		this.y = y;
	}



	public double getF() {
		return f;
	}



	public void setF(double f) {
		this.f = f;
	}



	@Override
	public String toString() {
		return "Measure [x=" + x + ", y=" + y + ", f=" + f + "]";
	}
	
}
