package hr.rasus.tests;

import java.util.Scanner;

import org.junit.Test;

import hr.fer.rasus.utils.Lock;

public class ThreadSync {
	Lock lock = new Lock();
	
	Runnable job = new Runnable() {
		@Override
		public void run() {
			lock.block();
			
		}
	};

	
	@Test
	public void test() {
		Thread t = new Thread(job);
		t.start();
		new Thread(job).start();
		
		try(Scanner s = new Scanner(System.in)){
			s.nextLine();
			System.out.println("SENDING: " + Thread.currentThread());
			lock.release();
			
			s.nextLine();
			System.out.println("SENDING: " + Thread.currentThread());
			lock.release();
		}
		
		try {
			t.join();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("Q");
		
	}
	
}
