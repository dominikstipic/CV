package hr.fer.zemris.java.hw07.utils;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Utility methods used by commands 
 * @author Dominik Stipić
 *
 */
public class Util {
	
	/**
	 * Tries to make apsolute path from given relative path
	 * @param parent current apsolute path
	 * @param child path relative to shell's current path 
	 * @return resolved path or null if can't build new path
	 */
	public static Path relativize(Path parent, Path child) {
		if(!child.isAbsolute()) {
			child = parent.resolve(child).normalize();
			if(Files.notExists(child)) {
				return null;
			}
		}
		return child;
	}
}
