package hr.fer.zemris.tests;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static java.lang.Math.cos;
import static java.lang.Math.sin;
import static java.lang.Math.toRadians;
import static java.lang.Math.sqrt;

import org.junit.Test;

import hr.fer.zemris.math.Vector2D;

public class forVectors {

	Vector2D vectors[] = {
			new Vector2D(1, 2),
			new Vector2D(-1, 84.3),
			new Vector2D(-53.55, -1.235),
			new Vector2D(12, -1.123),
			new Vector2D(0.00000000000000000002, 0.000000000001)
	};
	
	@Test
	public void forTranslating() {
		Vector2D offset = new Vector2D(1, 1);
		
		assertEquals(new Vector2D(2, 3),vectors[0].translated(offset));
		assertEquals(new Vector2D(0, 85.3),vectors[1].translated(offset));
		assertEquals(new Vector2D(-52.55, -0.235),vectors[2].translated(offset));
		assertEquals(new Vector2D(13, -0.123),vectors[3].translated(offset));
		assertEquals(new Vector2D(1, 1),vectors[4].translated(offset));
		
		offset = new Vector2D(-1.5, -14);
		double x = -1.5;
		double y = -14;
		
		assertEquals(new Vector2D(1+x, 2+y),vectors[0].translated(offset));
		assertEquals(new Vector2D(-1+x, 84.3+y),vectors[1].translated(offset));
		assertEquals(new Vector2D(-53.55+x, -1.235+y),vectors[2].translated(offset));
		assertEquals(new Vector2D(12+x, -1.123+y),vectors[3].translated(offset));
		assertEquals(new Vector2D(0+x, 0+y),vectors[4].translated(offset));
	}
	
	@Test
	public void forRotating() {
		Vector2D rad = new Vector2D(1, 0); 
		int FULL_CIRCLE = 360;
		
		for(int i = 1 ; i < FULL_CIRCLE ; ++i) {
			rad.rotate(1);  //1 degree
			double x = cos(toRadians(i));
			double y = sin (toRadians(i));
			assertEquals(new Vector2D(x, y),rad);
			assertEquals(i,rad.getAngle(),Vector2D.ZERO_INTERVAL);
			System.out.println(rad.getAngle());
			
		}
		
		Vector2D radInv = new Vector2D(1, 0); 
		for(int i = -1 ; i >= -1*FULL_CIRCLE ; --i) {
			radInv.rotate(-1);  // -1 degree
			double x = cos(toRadians(i));
			double y = sin (toRadians(i));
			assertEquals(new Vector2D(x, y),radInv);
			assertEquals(360 + i,radInv.getAngle(),Vector2D.ZERO_INTERVAL);
			System.out.println(radInv.getAngle());
			
		}
	}
	
	@Test
	public void forScaling() {
		assertEquals(new Vector2D(1*2, 2*2),vectors[0].scaled(2));
		assertEquals(new Vector2D(-1*4, 84.3*4),vectors[1].scaled(4));
		assertEquals(new Vector2D(-53.55*3.2, -1.235*3.2),vectors[2].scaled(3.2));
		assertEquals(new Vector2D(12*1.2, -1.123*1.2),vectors[3].scaled(1.2));
		assertEquals(new Vector2D(0, 0),vectors[4].scaled(-4.33));
		assertEquals(new Vector2D(-4.3*6.23, 1.6*6.23),new Vector2D(4.3, -1.6).scaled(-6.23));
		
		
	}
	
	@Test
	public void forNormalization() {
		Vector2D vec0 = new Vector2D(3, 1);
		Vector2D vec1 = new Vector2D(-2, -2);
		Vector2D vec2 = new Vector2D(6.43, 87);
		
		vec0.normalize();
		vec1.normalize();
		vec2.normalize();
		
		assertEquals(new Vector2D(3/sqrt(10), 1/sqrt(10)),vec0);
		assertEquals(new Vector2D(-2/sqrt(8), -2/sqrt(8)),vec1);
		assertEquals(new Vector2D(6.43/sqrt(6.43*6.43 + 87*87), 87/sqrt(6.43*6.43 + 87*87)),vec2);
	}
}
