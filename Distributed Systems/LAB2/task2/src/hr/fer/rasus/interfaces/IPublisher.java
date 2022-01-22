package hr.fer.rasus.interfaces;

public interface IPublisher {
	boolean subscribe(ISubscriber subscriber);
	void unsubscribe();
	void notifySubscribers(String message);
}
