package hr.fer.zemris.java.crypto;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.MessageFormat;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;


/**
 * Allowes user to encrypt/decrypt and calculate digests from given file.
 * For encryption and decryption AES cryptoalgorith is used and for digest
 * calculation SHA-256 is used.
 * Argumets from outside are expected.Here are the list of possible arguments:<br>
 * <b>checksha [FILE]</b> -> calculates 256 bit long digest for given file.<br>
 * <b>encrypt [FILE_SOURCE] [FILE_DEST]</b> -> encrypts file and puts encrypted file in specified path<br>
 * <b>decrypt [FILE_SOURCE] [FILE_DEST]</b> -> decrypts file and puts decrypted file in specified path<br>
 * @author Dominik Stipić
 *
 */
public class Crypto {
	
	/**
	 * commands
	 */
	public static String command1 = "checksha";
	public static String command2 = "encrypt";
	public static String command3 = "decrypt";
	
	/**
	 * prompts for user
	 */
	public static MessageFormat prompt1 = new MessageFormat("Please provide expected sha-256 digest for {0}");
	public static String prompt2 = "Please provide password as hex-encoded text (16 bytes, i.e. 32 hex-digits):";
	public static String prompt3 = "Please provide initialization vector as hex-encoded text (32 hex-digits):";
	
	/**
	 * responses to user action
	 */
	public static MessageFormat response1 = new MessageFormat("Digest of {0} {1} expected digest{2}");
	public static MessageFormat response2 = new MessageFormat("{0} completed. Generated file {1} based on file {2}.");
	
	
	/**
	 * Method which is automaticaly started when program is runned
	 * @param args argumets provided from command line interface
	 */
	public static void main(String[] args) {
		if(args.length != 2 && args.length != 3) {
			System.out.println("Invalid number of arguments");
			System.exit(1);
		}
		Path file = Paths.get(args[1]);
		if(Files.notExists(file)) {
			System.out.println("cannot find file with specified path - " + file);
			System.exit(1);
		}
		
		if(args[0].equals(Crypto.command1)) {
			if(args.length != 2) {
				System.out.println("checkSha command expects 2 argumets");
				System.exit(1);
			}
			try {
				String digest = userInput(Crypto.prompt1.format(new Object[] {file.getFileName()})).get(0);
				checkDigest(digest,file);
			} catch (Exception e ) {
				System.out.println(e.getMessage());
				System.exit(1);
			}
		}
		else if(args[0].equals(Crypto.command2)) {
			if(args.length != 3) {
				System.out.println("Encrypt command expects 3 argumets");
				System.exit(1);
			}
			Path encryptedFile = Paths.get(args[2]);
			List<String> passwordAndVector = userInput(Crypto.prompt2,Crypto.prompt3);
			AES_Crypto aes = new AES_Crypto(passwordAndVector.get(0), passwordAndVector.get(1));
			try {
				aes.process(file, encryptedFile, true);
				System.out.println(Crypto.response2.format(new Object[]{"Encryption", file, encryptedFile}));
			} catch (Exception e) {
				System.out.println(e.getMessage());
				System.exit(1);
			} 
		}
		else if(args[0].equals(Crypto.command3)) {
			if(args.length != 3) {
				System.out.println("Decrypt command expects 3 argumets");
				System.exit(1);
			}
			Path decryptedFile = Paths.get(args[2]);
			List<String> passwordAndVector = userInput(Crypto.prompt2,Crypto.prompt3);
			try {
				AES_Crypto aes = new AES_Crypto(passwordAndVector.get(0), passwordAndVector.get(1));
				aes.process(file, decryptedFile, false);
				System.out.println(Crypto.response2.format(new Object[]{"Decryption",decryptedFile ,file}));
			} catch (Exception e) {
				System.out.println(e.getMessage());
				System.exit(1);
			} 
		}
		else {
			throw new IllegalArgumentException("Invalid command provided");
		}
	}

	/**
	 * Takes user input
	 * @param prompts- messages which will be showed to user
	 * @return list of user inputs
	 */
	public static List<String> userInput(String ...prompts) {
		List<String> list = new LinkedList<>(); 
		Scanner s = new Scanner(System.in);
		for(String prompt:prompts) {
			System.out.println(prompt);
			System.out.println(">");
			String input = s.nextLine().trim();
			list.add(input);
		}
		s.close();
		return list;
	}
	
	/**
	 * Cheks if the inputed digest matches calculated with SHA algorithm
	 * @param inputDigest - user inputed digest
	 * @param file - file from which digest will be calculated
	 * @throws IOException - if error using files occurs
	 */
	public static void checkDigest(String inputDigest,Path file) throws IOException {
		SHA_Digest digestChecker = new SHA_Digest(file);
		try {
			boolean value = digestChecker.compareDigests(Util.hexToByte(inputDigest));
			System.out.println("Digesting completed.");
			if(value == true) {
				System.out.println(Crypto.response1.format(new Object[]{
						file,
						"matches",""}));
			}
			else {
				System.out.println(Crypto.response1.format(new Object[]{
						file,
						"does not match the",
						".Digest was:\n" + Util.byteToHex(digestChecker.getCaluatedDigest())}));
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println(e.getMessage());
			System.exit(2);
		}
		
		
	}
}
