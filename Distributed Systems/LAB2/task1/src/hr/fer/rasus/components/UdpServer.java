package hr.fer.rasus.components;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;

import hr.fer.rasus.interfaces.IPublisher;
import hr.fer.rasus.interfaces.ISubscriber;
import hr.fer.rasus.utils.LogService;
import hr.fer.rasus.utils.Protocol;

public class UdpServer implements Runnable, IPublisher<UdpServer>{
	private DatagramSocket socket;
	private boolean running;
	private byte[] buf = new byte[2048];
	private String name;
	private ISubscriber subscriber;
	
	public UdpServer(String name, int port) throws SocketException {
		socket = new DatagramSocket(port);
		this.name = name;
	}
	
	@Override
	public void run() {
		running = true;
		while(running) {
			DatagramPacket packet = new DatagramPacket(buf, buf.length);
			try {
				socket.receive(packet);
			} catch (IOException e) {
				e.printStackTrace();
			}
			String rcvString = new String(packet.getData(), packet.getOffset(), packet.getLength());
			String log = String.format("SERVER %s: %s\n", name, rcvString);
			LogService.get().print(log);
			messageHandler(rcvString);
		}
	}
	
	private void messageHandler(String rcvString) {
		if(Protocol.isEnd(rcvString)) {
			running = false;
		}
		else if (Protocol.isShare(rcvString)) {
		}
		else {
			// Everything else which isn't protocol message is printed on stdout
			System.out.println("UNRECOGNIZED " + rcvString);
		}
		
		notifySubscribers(rcvString);
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
		subscriber.update(message);
	}
}
