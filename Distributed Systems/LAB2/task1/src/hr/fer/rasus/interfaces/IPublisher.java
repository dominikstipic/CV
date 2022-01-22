package hr.fer.rasus.interfaces;

public interface IPublisher<T> {
	boolean subscribe(ISubscriber subscriber);
	void unsubscribe();
	void notifySubscribers(String message);
}
