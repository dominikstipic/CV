package hr.fer.zemris.java.hw06.demo2;

import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * Collection which holdes certain amount of prime numbers 
 * @author Dominik Stipic
 *
 */
public class PrimesCollection implements Iterable<Integer>{
	/**
	 * number of prime numbers which this collection holds
	 */
	private int number;

	/**
	 * Constructs collection which stores specified amoun of primes
	 * @param number of primes which will be storred in this collection
	 */
	public PrimesCollection(int number) {
		if(number < 0 ) {
			throw new IllegalArgumentException("number of primes must be positive");
		}
		this.number = number;
	}

	@Override
	public Iterator<Integer> iterator() {
		return new IteratorImpl();
	}
	
	/**
	 * Implementation of iterator which knows how to iterate over elements of this collection
	 * @author Dominik Stipic
	 *
	 */
	private class IteratorImpl implements Iterator<Integer>{
		/**
		 * index of prime 
		 */
		int count = 0;
		/**
		 * current prime number
		 */
		int prime = 2;
		
		@Override
		public boolean hasNext() {
			return count < number;
		}

		@Override
		public Integer next() {
			if(count >= number) {
				throw new NoSuchElementException("There isn't any prime number left");
			}
			if(count == 0) {
				++count;
				return prime = 2;
			}
			while(true) {
				++prime;
				boolean flag = true;
				for(int i = 2; i < prime; ++i) {
					if(prime % i == 0) {
						flag = false;
						break;
					}
				}
				if(flag == true) {
					++count;
					return prime;
				}
			} 
		}
		
	}
}
