package hr.fer.rasus.utils;

public class Lock {
	private Object monitor  = new Object();
	

	public void block() {
		synchronized (monitor) {
			try {
				System.out.println("MONITOR: " + Thread.currentThread());
				monitor.wait();
				System.out.println("EXIT: " + Thread.currentThread());
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	public void release() {
		synchronized (monitor) {
			monitor.notify();
		}
	}
	
}
