package hr.fer.rasus.components;

import java.util.Random;

import hr.fer.rasus.interfaces.IPublisher;
import hr.fer.rasus.interfaces.ISubscriber;
import hr.fer.rasus.utils.Protocol;
import hr.fer.rasus.utils.Utils;

public class TriggerClock implements Runnable, IPublisher<TriggerClock>{
    private long startTime;
    private double jitter;
    private long triggerPeriod;
    private ISubscriber subscriber;
    private boolean running;
    public final int INCREMENT_CONSTANT;
    private Long logicalClock = 0l;

    public TriggerClock(long period) {
        startTime = System.currentTimeMillis();
        Random r = new Random();
        jitter = (r.nextInt(400) - 200) / 1000d;
        triggerPeriod = period;
        INCREMENT_CONSTANT = Utils.getRandomInt(1, 7);
    }
    
    public long currentTimeMillis() {
        long current = System.currentTimeMillis();
        long diff = current - startTime;
        double coef = diff / 1000;
        return startTime + Math.round(diff * Math.pow((1+jitter), coef));
    }
    
    public long activeMilli() {
    	long currentTime = currentTimeMillis();
    	return currentTime - startTime;
    }
    
    public int activeSec() {
    	int time = (int) activeMilli();
    	int seconds = time/1000;
    	return seconds;
    }

	@Override
	public void run() {
		running = true;
		while(running) {
			Long periodStart = currentTimeMillis();
			try {
				Thread.sleep(triggerPeriod);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			Long periodEnd = currentTimeMillis();
			String message = Protocol.triggerMessage(periodStart, periodEnd);
			notifySubscribers(message);
		}
	}
	
	@Override
	public boolean subscribe(ISubscriber subscriber) {
		if(this.subscriber != null) return false;
		this.subscriber = subscriber;
		return true;
	}

	@Override
	public void unsubscribe() {
		this.subscriber = null;
	}

	@Override
	public void notifySubscribers(String message) {
		if(subscriber == null)return;
		subscriber.update(message);
	}
	
	public void turnOff() {
		running = false;
	}
	
	public Long getLogicalClock() {
		return logicalClock;
	}
	
	public void incLogicalClock() {
		logicalClock += INCREMENT_CONSTANT;
	}
	
	public void incLogicalClock(Long increment) {
		logicalClock += increment;
	}
	
}