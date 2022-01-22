/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package hr.fer.rasus.components;
import java.util.Random;

import hr.fer.rasus.interfaces.IPublisher;
import hr.fer.rasus.interfaces.ISubscriber;
import hr.fer.rasus.utils.Protocol;

/**
 *
 * @author Aleksandar
 */
public class EmulatedSystemClock implements Runnable, IPublisher{
    
    private long startTime;
    private double jitter; //jitter per second,  percentage of deviation per 1 second
    private Long triggerPeriod;
    private ISubscriber subscriber;
    private boolean running;

    public EmulatedSystemClock(Long triggerPeriod) {
    	this.triggerPeriod = triggerPeriod;
        startTime = System.currentTimeMillis();
        Random r = new Random();
        jitter = (r.nextInt(400) - 200) / 1000d; //divide by 10 to get the interval between [-20, 20], and then divide by 100 to get percentage
    }
    
    private long currentTimeMillis() {
        long current = System.currentTimeMillis();
        long diff =current - startTime;
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

    
	public void turnOff() {
		running = false;
	}
	
    
}
