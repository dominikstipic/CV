package hr.fer.zemris.java.crypto;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.spec.AlgorithmParameterSpec;

import javax.crypto.BadPaddingException;
import javax.crypto.Cipher;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;

/**
 * Represents Aes cripto algorithm which offers methods for encryption and decryption of some file.
 * In order that decryption and encryption is completed succesfull, 
 * cipher key text and initializaton vector is needed.  
 * @author Dominik Stipić
 * 
 */
public class AES_Crypto {
	/**
	 * cipher algorithm
	 */
	private Cipher cipher;
	/**
	 * Secret key 
	 */
	private SecretKeySpec keySpec;
	/**
	 * Algorithm parametars
	 */
	private AlgorithmParameterSpec paramSpec;

	/**
	 * Creates and sets up enviroment for encryption/decryption
	 * @param keyText secret key text
	 * @param ivText initialization vector
	 */
	public AES_Crypto(String keyText, String ivText) {
		keySpec = new SecretKeySpec(Util.hexToByte(keyText), "AES");
		paramSpec = new IvParameterSpec(Util.hexToByte(ivText));
		try {
			cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
		} catch (Exception wontHappen) {
		}
	}

	/**
	 * Does encryption/decryption
	 * @param source File which will be processed
	 * @param destination destianton of processed file
	 * @param encrypt - if true -> encryption will be applied
	 * 					if false -> decryption will be applied
	 * @throws InvalidKeyException - invalid key provided
	 * @throws InvalidAlgorithmParameterException - error in algorithm specification
	 * @throws IOException - if error using files occurs
	 * @throws IllegalBlockSizeException - illelag block size occured
	 * @throws BadPaddingException - bad padding occured
	 */
	public void process(Path source, Path destination, boolean encrypt) throws InvalidKeyException,
			InvalidAlgorithmParameterException, IOException, IllegalBlockSizeException, BadPaddingException {
		
		try (BufferedInputStream is = new BufferedInputStream(Files.newInputStream(source));
				BufferedOutputStream os = new BufferedOutputStream(Files.newOutputStream(destination))) {
			
			cipher.init(encrypt ? Cipher.ENCRYPT_MODE : Cipher.DECRYPT_MODE, keySpec, paramSpec);
			byte[] buff = new byte[1024];
			while (true) {
				int count = is.read(buff);
				if (count == -1)
					break;
				byte[] b = cipher.update(buff, 0, count);
				os.write(b);
			}
			byte[] b = cipher.doFinal();
			os.write(b);
		}
	}

}
