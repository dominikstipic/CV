package hr.fer.nenr.interfaces;

public interface IPublisher {
	void subscribe(ISubscriber subscriber);
	void unsubscribe(ISubscriber subscriber);
	void notifySubscribers(Object object);
}
