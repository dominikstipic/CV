package hr.fer.zemris.java.crypto;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;

/**
 * Calculates 256 bit SHA_digest for given file and then
 * comapares calculated digest with digest given by user.
 * @author Dominik Stipić
 *
 */
public class SHA_Digest {
	/**
	 * file for which sha is calculated 
	 */
	private Path file;
	/**
	 * Sha digest calcualtion machine 
	 */
	MessageDigest sha;
	/**
	 * Used digest algorithm
	 */
	public static final String algorithm = "SHA-256";
	/**
	 * calculated digest
	 */
	private byte[] digest;
	
	/**
	 * Creates SHA_digest calculation machine
	 * @param file - file for which sha is calculated 
	 * @throws IOException - if error using files occurs
	 */
	public SHA_Digest(Path file) throws IOException  {
		this.file = file;
		try {
			sha = MessageDigest.getInstance(algorithm);
		} catch (NoSuchAlgorithmException ignorable) {}
	}
	
	/**
	 * Compares calcualted digest and digest provided by user
	 * @param digestByte digest provided by user
	 * @return apporpriate boolean value
	 * @throws IOException if error using files occurs
	 */
	public boolean compareDigests(byte[] digestByte) throws IOException {
		try(BufferedInputStream is = new BufferedInputStream(Files.newInputStream(file))){
			byte[] buff = new byte[1024];
			while(true) {
				int read = is.read(buff);
				if(read == -1)break;
				sha.update(buff, 0 ,read);
			}
			digest = sha.digest();
			return Arrays.equals(digest, digestByte);
		}
	}
	
	/**
	 * gets calculated digest
	 * @return calculated digest
	 */
	public byte[] getCaluatedDigest() {
		return digest;
	}
	
	
	
	
	
}
