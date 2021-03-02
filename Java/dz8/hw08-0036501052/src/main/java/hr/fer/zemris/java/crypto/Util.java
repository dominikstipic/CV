package hr.fer.zemris.java.crypto;

import java.util.Objects;

/**
 * Provides static methods for tranforming hexadecimal represenation into bytes and vice verse.
 * @author Dominik Stipić
 *
 */
public class Util {
	
	/**
	 * Converts hexadecimal represntation into array of bytes
	 * @param keyText hexadecimal representation
	 * @return byte array
	 */
	public static byte[] hexToByte(String keyText) {
		Objects.requireNonNull(keyText);
		if(keyText.length() % 2 == 1 ) {
			throw new IllegalArgumentException("The String can't be converted to bytes - odd number of characters");
		}
		if (keyText.length() == 0) {
			return new byte[0];
		}
		
		byte[] bin = new byte[keyText.length() / 2];
		int byteIndex = 0;
		for(int i = 0; i < keyText.length() ; i = i+2) {
			byte b1 = (byte)(Character.digit(keyText.charAt(i), 16));
			byte b2 = (byte)(Character.digit(keyText.charAt(i+1), 16));	
			if(b1 == -1 || b2 == -1 ) {
				throw new IllegalArgumentException("The String can't be converted to bytes - invalid hex symbol");
			}
			bin[byteIndex++] = (byte)(b1 << 4 | b2); 
		}
		return bin;
	}
	
	/**
	 * Converts array of bytes into hexadecimal represntation 
	 * @param bytes - array of bytes 
	 * @return hexadecimal represntation of bytes
	 */
	public static String byteToHex(byte [] bytes) {
		Objects.requireNonNull(bytes);
		StringBuilder hex = new StringBuilder();
		
		for(byte by:bytes) {
			hex.append(String.format("%02x", by));
		}
		return hex.toString();
	}
}
