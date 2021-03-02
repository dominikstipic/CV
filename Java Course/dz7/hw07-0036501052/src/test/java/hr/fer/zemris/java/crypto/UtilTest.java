package hr.fer.zemris.java.crypto;

import org.junit.Assert;
import org.junit.Test;


public class UtilTest {
	
	@Test
	public void forHexToBytes() {
		String h1 = "01aE22";
		String h2 = "ffff";
		String h3 = "F101e1";
		String h4 = "000001";
		String h5 = "";
		
		Assert.assertArrayEquals(new byte[] {1, -82, 34},  Util.hexToByte(h1));
		Assert.assertArrayEquals(new byte[] {-1, -1}, Util.hexToByte(h2));
		Assert.assertArrayEquals(new byte[] {-15, 1, -31}, Util.hexToByte(h3));
		Assert.assertArrayEquals(new byte[] {0, 0, 1}, Util.hexToByte(h4));
		Assert.assertArrayEquals(new byte[] {}, Util.hexToByte(h5));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void forHexToBytesFalseInput1() {
		String h1 = "01ar22";
		
		Util.hexToByte(h1);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void forHexToBytesFalseInput2() {
		String h1 = "01a";
		
		Util.hexToByte(h1);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void forHexToBytesFalseInput3() {
		String h1 = "01r";
		
		Util.hexToByte(h1);
	}
	
	
	@Test
	public void forBytesToHex() {
		Assert.assertEquals("01ae22" , Util.byteToHex(new byte[] {1, -82, 34}));
		Assert.assertEquals("01ff37", Util.byteToHex(new byte[] {1, -1, 55}));
		Assert.assertEquals("185a", Util.byteToHex(new byte[] {24, 90}));
		Assert.assertEquals("d4fe", Util.byteToHex(new byte[] {-44, -2}));
		Assert.assertEquals("01040503", Util.byteToHex(new byte[] {1,4,5,3}));
		Assert.assertEquals("", Util.byteToHex(new byte[] {}));
	}
}
